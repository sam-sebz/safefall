"""Adds a person who can sign in, or changes their password.

Usage: python create_user.py <username>
"""
import getpass
import os
import sys

from flask import Flask
from werkzeug.security import generate_password_hash

from storage import User, db

MIN_PASSWORD_LENGTH = 8


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    username = sys.argv[1].strip()
    password = getpass.getpass("Password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    if password != getpass.getpass("Password again: "):
        sys.exit("The passwords don't match.")

    base = os.path.abspath(os.path.dirname(__file__))
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(base, 'instance', 'users.db')}"
    db.init_app(app)
    with app.app_context():
        db.create_all()
        user = User.query.filter_by(username=username).first()
        hashed = generate_password_hash(password, method="pbkdf2:sha256")
        if user:
            user.password = hashed
        else:
            db.session.add(User(username=username, password=hashed))
        db.session.commit()
    print(f"Saved {username}.")


if __name__ == "__main__":
    main()
