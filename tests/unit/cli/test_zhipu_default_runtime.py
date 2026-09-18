"""Test generated defaults through the actual SDK and vector length checks."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from cli.commands import onboard
from cli.lib.deploy_helpers import generate_quickstart_env
from cli.main import build_parser


@pytest.mark.parametrize("interactive", [False, True])
def test_zhipu_default_matches_native_embedding_response(tmp_path, monkeypatch, interactive):
    pytest.importorskip("openai")
    from providers.embedder.openai_embedder import OpenAIEmbedder

    args = build_parser().parse_args([
        "onboard", "--mode", "docker", "--non-interactive", "--provider", "zhipu", "--api-key", "DUMMY_TEST_KEY",
    ])
    llm = onboard._step2_llm(args)
    if interactive:
        args.non_interactive = False
        monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
        monkeypatch.setattr(onboard, "input_validated", lambda label, default: default)
    embedding = onboard._step3_embedding(args, llm)
    path = generate_quickstart_env(llm, embedding, output_path=tmp_path / "ogmemory.env")
    assert "export EMBEDDING_PROVIDER=openai" in path.read_text()
    assert "export EMBEDDING_DIMENSION=2048" in path.read_text()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"object": "list", "model": "embedding-3", "data": [
                {"object": "embedding", "index": 0, "embedding": [0.0] * 2048},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        client = OpenAIEmbedder(api_key="DUMMY_TEST_KEY",
            base_url=f"http://127.0.0.1:{server.server_port}/v4", model=embedding["model"],
            dimension=embedding["dimension"])
        vectors = client.embed_texts(["default dimension regression"])
        assert len(vectors[0]) == 2048
        assert requests[0]["model"] == "embedding-3"
        # Existing image/client behavior remains compatible; no new API option
        # is required, including for models that reject a dimensions argument.
        assert "dimensions" not in requests[0]
    finally:
        if client is not None:
            client._client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
