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
    "Your password expires in 24 hours. Update NOW: http://account-update.xyz/reset URGENT",
    "Congratulations! You are selected for $1000 Amazon voucher. http://amaz0n-offer.com FREE!!!",
    "IRS TAX REFUND: $849 pending. Confirm identity at http://irs-refund.biz/claim immediately.",
    "Your credit card is on HOLD. Remove hold here: http://172.16.0.1/creditcard/verify",
    "Security breach on your account. Re-enter credentials: http://secure-login.info NOW!!!",
    "Final warning: confirm your email or lose access http://confirm-account.biz URGENT ACT NOW",
    "Your Netflix subscription failed. Update payment: http://netflix-billing.xyz/update",
    "WINNER! iPhone 15 giveaway. Register at http://iphone-free.win today only! FREE offer!",
    "Your Barclays account will close. Verify details at http://barclays-secure.net/verify",
    "Click to claim your exclusive offer http://deals-now.biz 90% OFF limited time winner",
    "ACTION REQUIRED: login at http://194.23.5.1/ebay/confirm to avoid suspension URGENT",
] * 120

LEGIT = [
    "Hi Baljinder, just checking if you are free for a call on Thursday at 3pm?",
    "Your order #84729 has been shipped and will arrive by Friday. Track via your account.",
    "Monthly newsletter: Here are this month's top articles. We hope you enjoy reading.",
    "Thank you for attending yesterday's webinar. Slides and recording are now available.",
    "Reminder: your library book is due this Friday. Renew online through your account.",
    "Please find attached the invoice for March services. Let me know if you have questions.",
    "The team meeting has been moved to 2pm on Wednesday. Calendar invite updated.",
    "Congratulations on finishing the course! Your certificate is now in your profile.",
    "Your appointment is confirmed for Monday at 10am. Please arrive 10 minutes early.",
    "We have updated our privacy policy effective from 1st May. Please review at your convenience.",
    "Hi, following up on last week's proposal. Could we schedule a 30-minute call?",
    "Your package was delivered to your front door at 2:34pm. Tracking number: UPS8837291.",
    "February bank statement is now available. Download it from your online banking portal.",
    "Thank you for your feedback! We appreciate you taking the time to share your experience.",
    "Reminder: annual subscription renews on the 15th. No action needed unless cancelling.",
] * 120


def load_data(csv_path: str) -> tuple:
    """Load dataset; fall back to synthetic if CSV not found."""
    if os.path.exists(csv_path):
        log.info("Loading Kaggle dataset from %s …", csv_path)
        df = pd.read_csv(csv_path)

        text_col  = next((c for c in ["text_combined","text","Email Text","body"] if c in df.columns), df.columns[0])
        label_col = next((c for c in ["label","Label","spam","is_phishing"] if c in df.columns), df.columns[-1])

        df = df[[text_col, label_col]].dropna()
        df.columns = ["text", "label"]
        df["label"] = df["label"].astype(int)
        log.info("Loaded %d rows (%d phish / %d legit)",
                 len(df), (df.label==1).sum(), (df.label==0).sum())
        return df["text"].tolist(), df["label"].tolist()

    log.warning("Kaggle CSV not found — using synthetic fallback dataset.")
    log.warning("Download from: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset")
    texts  = PHISH + LEGIT
    labels = [1]*len(PHISH) + [0]*len(LEGIT)
    import random; random.seed(42)
    combined = list(zip(texts, labels)); random.shuffle(combined)
    texts, labels = zip(*combined)
    log.info("Synthetic dataset: %d samples", len(texts))
    return list(texts), list(labels)


def main():
    parser = argparse.ArgumentParser(description="Train phishing detector")
    parser.add_argument("--csv", default=KAGGLE_CSV, help="Path to dataset CSV")
    args = parser.parse_args()

    init_db()
    texts, labels = load_data(args.csv)

    model   = PhishingDetector()
    metrics = model.fit(texts, labels)
    model.save()
    save_metrics(metrics, len(texts))

    print("\n" + "="*50)
    print("  TRAINING COMPLETE")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
    print("="*50 + "\n")
    print("  Next step:")
    print('  python analyse.py --text "Your urgent email here"')
    print('  python analyse.py --file path/to/email.txt\n')


if __name__ == "__main__":
    main()
