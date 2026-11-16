"""
set_admin_password.py
-----------------------
Manages login accounts for the Flask webview (app.py). Supports
MULTIPLE admin/guard accounts - run this any time to add a new
account or change an existing one's password.

Passwords are stored as hashes (never in plain text) in
data/admin_credentials.json.

NOTE: this version shows the password as you type it (no hidden/dot
masking), since some terminals (especially certain Windows setups)
don't support hidden input reliably. Make sure no one else can see
your screen while you run this.

Usage:
    python set_admin_password.py
"""

import os
import json
from werkzeug.security import generate_password_hash

CREDENTIALS_FILE = "data/admin_credentials.json"


def load_admins():
    """Returns the list of admin accounts, migrating the old single-account format if needed."""
    if not os.path.exists(CREDENTIALS_FILE):
        return []

    with open(CREDENTIALS_FILE, "r") as f:
        data = json.load(f)

    if "admins" in data:
        return data["admins"]

    # Old format: {"username": ..., "password_hash": ...} - migrate to a list.
    if "username" in data and "password_hash" in data:
        return [{"username": data["username"], "password_hash": data["password_hash"]}]

    return []


def save_admins(admins):
    os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump({"admins": admins}, f, indent=2)


def set_password():
    admins = load_admins()

    if admins:
        print("Current admin accounts:")
        for a in admins:
            print(f"  - {a['username']}")
        print()

    username = input("Username to add or update: ").strip()
    password = input("Password (will be visible as you type): ").strip()
    confirm = input("Confirm password: ").strip()

    if not username or not password:
        print("Username and password cannot be empty.")
        return

    if password != confirm:
        print("Passwords did not match. Try again.")
        return

    password_hash = generate_password_hash(password)
    existing = next((a for a in admins if a["username"] == username), None)

    if existing:
        existing["password_hash"] = password_hash
        print(f"\nPassword updated for existing account '{username}'.")
    else:
        admins.append({"username": username, "password_hash": password_hash})
        print(f"\nNew admin account '{username}' added.")

    save_admins(admins)
    print(f"Total admin accounts: {len(admins)}")


if __name__ == "__main__":
    set_password()
