"""为指定用户签发服务身份令牌（X-Auth-Token）。

用法：
    python scripts/mint_token.py 3                # 默认 7 天
    python scripts/mint_token.py 3 --ttl 3600     # 1 小时
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.auth import sign_user_token  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="签发 X-Auth-Token")
    parser.add_argument("user_id", type=int, help="users 表里的用户 ID")
    parser.add_argument("--ttl", type=int, default=7 * 24 * 3600, help="有效期（秒），默认 7 天")
    args = parser.parse_args()

    token = sign_user_token(args.user_id, ttl_seconds=args.ttl)
    print(f"user_id={args.user_id} ttl={args.ttl}s")
    print(f"X-Auth-Token: {token}")
    print("调用示例：")
    print(f'  curl -H "X-Auth-Token: {token}" http://localhost:8000/api/v1/issues')


if __name__ == "__main__":
    main()
