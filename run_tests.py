#!/usr/bin/env python3
"""测试运行脚本 - 自动设置测试环境"""

import os
import sys
import subprocess
from pathlib import Path


def setup_test_env():
    """设置测试环境变量"""
    env = {
        "TEST_DB_HOST": os.environ.get("TEST_DB_HOST", "localhost"),
        "TEST_DB_PORT": os.environ.get("TEST_DB_PORT", "5432"),
        "TEST_DB_NAME": os.environ.get("TEST_DB_NAME", "test_memory_db"),
        "TEST_DB_USER": os.environ.get("TEST_DB_USER", "postgres"),
        "TEST_DB_PASSWORD": os.environ.get("TEST_DB_PASSWORD", "test_password"),
    }

    missing_keys = []
    for key in ["OPENAI_API_KEY"]:
        if not os.environ.get(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"警告: 缺少环境变量: {', '.join(missing_keys)}")
        print("请设置以下环境变量:")
        print("  export OPENAI_API_KEY=your_api_key")
        print("\n或者使用测试模式（跳过需要API密钥的测试）:")
        print("  python run_tests.py --skip-api")
        return False

    return True


def run_tests(args):
    """运行测试"""
    pytest_args = ["python", "-m", "pytest", "tests/"]

    if args.verbose:
        pytest_args.append("-v")

    if args.coverage:
        pytest_args.extend(["--cov=src", "--cov-report=term-missing"])

    if args.unit_only:
        pytest_args.append("tests/unit/")
    elif args.integration_only:
        pytest_args.append("tests/integration/")

    if args.marker:
        pytest_args.extend(["-m", args.marker])

    if args.skip_api:
        pytest_args.append("-m")
        pytest_args.append("not requires_openai")

    if args.file:
        pytest_args.append(args.file)

    print(f"运行测试: {' '.join(pytest_args)}")
    return subprocess.run(pytest_args)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="oG-Memory 测试运行器")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="生成测试覆盖率报告"
    )
    parser.add_argument("-u", "--unit-only", action="store_true", help="只运行单元测试")
    parser.add_argument(
        "-i", "--integration-only", action="store_true", help="只运行集成测试"
    )
    parser.add_argument("-m", "--marker", help="使用测试标记")
    parser.add_argument(
        "--skip-api", action="store_true", help="跳过需要 OpenAI API 的测试"
    )
    parser.add_argument("-f", "--file", help="运行特定的测试文件")

    args = parser.parse_args()

    if not setup_test_env():
        sys.exit(1)

    result = run_tests(args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
