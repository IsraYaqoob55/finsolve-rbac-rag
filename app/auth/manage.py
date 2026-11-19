"""User administration from the command line.

python -m app.auth.manage seed                      # create/refresh demo users
python -m app.auth.manage list
python -m app.auth.manage create jdoe "John Doe" finance   # prompts for password
python -m app.auth.manage set-role jdoe marketing
python -m app.auth.manage disable jdoe
python -m app.auth.manage enable jdoe
"""

import argparse
import getpass
import sys

import yaml

from app.auth.user_store import UserAlreadyExistsError, UserStore
from app.core.config import Settings, get_settings
from app.rbac.policy import RolePolicy


def _require_role(policy: RolePolicy, role: str) -> None:
    if role not in policy.roles:
        sys.exit(f"Unknown role '{role}'. Valid roles: {', '.join(policy.roles)}")


def seed(store: UserStore, policy: RolePolicy, settings: Settings) -> None:
    if settings.seed_user_password is None:
        sys.exit("Set SEED_USER_PASSWORD in .env first.")
    password = settings.seed_user_password.get_secret_value()
    raw = yaml.safe_load(settings.resolve(settings.seed_users_config).read_text(encoding="utf-8"))
    for entry in raw["users"]:
        _require_role(policy, entry["role"])
        store.upsert(entry["username"], entry["full_name"], entry["role"], password)
        print(f"  seeded {entry['username']:12} role={entry['role']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage chatbot users.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="create or refresh the demo users")
    sub.add_parser("list", help="list users")
    create = sub.add_parser("create", help="create a user (prompts for password)")
    create.add_argument("username")
    create.add_argument("full_name")
    create.add_argument("role")
    set_role = sub.add_parser("set-role", help="change a user's role")
    set_role.add_argument("username")
    set_role.add_argument("role")
    for name in ("disable", "enable"):
        sub.add_parser(name, help=f"{name} a user").add_argument("username")
    args = parser.parse_args()

    settings = get_settings()
    policy = RolePolicy.from_yaml(settings.resolve(settings.roles_config))
    store = UserStore(settings.resolve(settings.user_db_path))
    try:
        if args.command == "seed":
            seed(store, policy, settings)
        elif args.command == "list":
            for user in store.list_users():
                status = "DISABLED" if user.disabled else "active"
                print(f"{user.username:14} {user.role:12} {status:9} {user.full_name}")
        elif args.command == "create":
            _require_role(policy, args.role)
            password = getpass.getpass("Password: ")
            if password != getpass.getpass("Repeat password: "):
                sys.exit("Passwords do not match.")
            try:
                store.create(args.username, args.full_name, args.role, password)
            except (UserAlreadyExistsError, ValueError) as exc:
                sys.exit(f"Could not create user: {exc}")
            print(f"Created {args.username} ({args.role})")
        elif args.command == "set-role":
            _require_role(policy, args.role)
            store.set_role(args.username, args.role)
            print(f"{args.username} is now {args.role}")
        else:
            store.set_disabled(args.username, disabled=args.command == "disable")
            print(f"{args.username} {args.command}d")
    finally:
        store.close()


if __name__ == "__main__":
    main()
