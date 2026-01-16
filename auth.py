"""
Authentication module for JSON-based user management
"""
import json
import os
from typing import Optional, Dict, Any

USER_FILE = "users.json"

def load_users() -> Dict[str, Any]:
    """Load users from JSON file"""
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    return {"users": []}

def save_users(users_data: Dict[str, Any]) -> None:
    """Save users to JSON file"""
    with open(USER_FILE, 'w') as f:
        json.dump(users_data, f, indent=2)

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user credentials"""
    users_data = load_users()
    for user in users_data.get("users", []):
        if user.get("username") == username and user.get("password") == password:
            return True
    return False

def add_user(username: str, password: str, email: str = "") -> bool:
    """Add a new user to the system"""
    users_data = load_users()
    # Check if user already exists
    for user in users_data.get("users", []):
        if user.get("username") == username:
            return False
    
    users_data["users"].append({
        "username": username,
        "password": password,
        "email": email
    })
    save_users(users_data)
    return True

def get_user_info(username: str) -> Optional[Dict[str, Any]]:
    """Get user information by username"""
    users_data = load_users()
    for user in users_data.get("users", []):
        if user.get("username") == username:
            return user
    return None
