#!/usr/bin/env python3
"""
GitHub Repository Access Request System
Automatically invites users to private repository with read-only access
"""

import os
import sqlite3
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Your GitHub personal access token
GITHUB_OWNER = os.getenv('GITHUB_OWNER', 'yourusername')  # Your GitHub username
GITHUB_REPO = os.getenv('GITHUB_REPO', 'PCA_Dashboard')  # Repository name
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER')  # Your email
EMAIL_PASS = os.getenv('EMAIL_PASS')  # App password
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'trammellwhitfield@investalogical.com')

def init_database():
    """Initialize SQLite database for tracking requests"""
    conn = sqlite3.connect('access_requests.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            github_username TEXT NOT NULL,
            company TEXT,
            use_case TEXT NOT NULL,
            message TEXT,
            request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            github_invitation_sent BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

def send_github_invitation(github_username):
    """Send GitHub repository invitation with read permissions"""
    if not GITHUB_TOKEN:
        return False, "GitHub token not configured"
    
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    data = {
        'permission': 'pull'  # Read-only access
    }
    
    url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/collaborators/{github_username}'
    
    try:
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code == 201:
            return True, "Invitation sent successfully"
        elif response.status_code == 204:
            return True, "User already has access"
        else:
            return False, f"GitHub API error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return False, f"Error sending invitation: {str(e)}"

def send_notification_email(user_data, success, error_msg=None):
    """Send email notification to user and admin"""
    if not all([EMAIL_USER, EMAIL_PASS]):
        return False
    
    try:
        # Email to user
        user_msg = MIMEMultipart()
        user_msg['From'] = EMAIL_USER
        user_msg['To'] = user_data['email']
        user_msg['Subject'] = "PCA Dashboard Access Request - Investalogical"
        
        if success:
            user_body = f"""
Dear {user_data['name']},

Thank you for requesting access to the Investalogical PCA Stock Analysis Dashboard!

✅ Your GitHub repository invitation has been sent to: {user_data['github_username']}

What's Next:
1. Check your GitHub notifications (email or GitHub.com)
2. Accept the repository invitation 
3. You'll have read-only access to clone and view the repository
4. Follow the README instructions to set up the dashboard

Repository: https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}

If you have any questions or need assistance, please reply to this email.

Best regards,
The Investalogical Team
trammellwhitfield@investalogical.com
            """
        else:
            user_body = f"""
Dear {user_data['name']},

Thank you for your interest in the Investalogical PCA Stock Analysis Dashboard.

We received your access request, but there was an issue processing your GitHub invitation:
{error_msg}

Our team will review your request manually and contact you within 24 hours.

Best regards,
The Investalogical Team
trammellwhitfield@investalogical.com
            """
        
        user_msg.attach(MIMEText(user_body, 'plain'))
        
        # Email to admin
        admin_msg = MIMEMultipart()
        admin_msg['From'] = EMAIL_USER
        admin_msg['To'] = ADMIN_EMAIL
        admin_msg['Subject'] = f"New Repository Access Request - {user_data['name']}"
        
        admin_body = f"""
New access request received:

Name: {user_data['name']}
Email: {user_data['email']}
GitHub: {user_data['github_username']}
Company: {user_data.get('company', 'Not specified')}
Use Case: {user_data['use_case']}
Message: {user_data.get('message', 'None')}

Status: {'✅ Invitation sent successfully' if success else f'❌ Error: {error_msg}'}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        admin_msg.attach(MIMEText(admin_body, 'plain'))
        
        # Send emails
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(user_msg)
        server.send_message(admin_msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Email error: {e}")
        return False

@app.route('/')
def signup_form():
    """Display the signup form"""
    return render_template('signup.html')

@app.route('/request-access', methods=['POST'])
def request_access():
    """Process access request"""
    try:
        # Get form data
        user_data = {
            'name': request.form['name'],
            'email': request.form['email'],
            'github_username': request.form['github_username'],
            'company': request.form.get('company', ''),
            'use_case': request.form['use_case'],
            'message': request.form.get('message', '')
        }
        
        # Validate required fields
        if not all([user_data['name'], user_data['email'], user_data['github_username'], user_data['use_case']]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Save to database
        conn = sqlite3.connect('access_requests.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_requests 
            (name, email, github_username, company, use_case, message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_data['name'],
            user_data['email'], 
            user_data['github_username'],
            user_data['company'],
            user_data['use_case'],
            user_data['message']
        ))
        conn.commit()
        conn.close()
        
        # Send GitHub invitation
        success, message = send_github_invitation(user_data['github_username'])
        
        # Update database with invitation status
        conn = sqlite3.connect('access_requests.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE access_requests 
            SET status = ?, github_invitation_sent = ?
            WHERE email = ? AND github_username = ?
        ''', (
            'completed' if success else 'manual_review',
            success,
            user_data['email'],
            user_data['github_username']
        ))
        conn.commit()
        conn.close()
        
        # Send notification emails
        send_notification_email(user_data, success, message if not success else None)
        
        # Return success page
        return render_template('success.html', 
                             success=success, 
                             github_username=user_data['github_username'],
                             message=message)
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/admin/requests')
def admin_requests():
    """View all access requests (basic admin panel)"""
    conn = sqlite3.connect('access_requests.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM access_requests 
        ORDER BY request_date DESC
    ''')
    requests = cursor.fetchall()
    conn.close()
    
    return render_template('admin.html', requests=requests)

if __name__ == '__main__':
    init_database()
    
    # Check configuration
    if not GITHUB_TOKEN:
        print("⚠️  Warning: GITHUB_TOKEN not set. GitHub invitations will fail.")
        print("   Set environment variable: export GITHUB_TOKEN=your_token_here")
    
    if not all([EMAIL_USER, EMAIL_PASS]):
        print("⚠️  Warning: Email credentials not set. Notifications will fail.")
        print("   Set: EMAIL_USER and EMAIL_PASS environment variables")
    
    print("🚀 Starting GitHub Access Request System...")
    print(f"📊 Repository: {GITHUB_OWNER}/{GITHUB_REPO}")
    print("🌐 Visit: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)