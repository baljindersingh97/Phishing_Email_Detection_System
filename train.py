import os, sys, argparse, logging
import pandas as pd
from detector import PhishingDetector, init_db, save_metrics, log

BASE      = os.path.dirname(os.path.abspath(__file__))
KAGGLE_CSV = os.path.join(BASE, "data", "phishing_email.csv")

PHISH = [
    "URGENT: Your PayPal account has been suspended. Click http://192.168.0.1/verify NOW!",
    "Dear customer, suspicious activity detected. Verify at http://secure-paypal-login.biz immediately.",
    "You have WON $5,000! Claim your FREE prize at http://prize-winner.net/claim. Limited time!",
    "ALERT: Your bank account requires immediate verification. http://10.0.0.5/bank-login",
    "Security breach on your account. Re-enter credentials: http://secure-login.info NOW!!!",
    "Final warning: confirm your email or lose access http://confirm-account.biz URGENT ACT NOW"
    "ACTION REQUIRED: login at http://194.23.5.1/ebay/confirm to avoid suspension URGENT",
] * 120

LEGIT = [
    "Hi Baljinder, just checking if you are free for a call on Thursday at 3pm?",
    "Your order #84729 has been shipped and will arrive by Friday. Track via your account.",
    "Your package was delivered to your front door at 2:34pm. Tracking number: UPS8837291.",
    "February bank statement is now available. Download it from your online banking portal.",
    "Thank you for your feedback! We appreciate you taking the time to share your experience.",
    "Reminder: annual subscription renews on the 15th. No action needed unless cancelling.",
] * 120
