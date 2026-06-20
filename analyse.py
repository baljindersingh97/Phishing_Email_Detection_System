import os, sys, argparse
from detector import PhishingDetector, extract_features, save_result, get_history, MODEL

RED   = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
BOLD  = "\033[1m";  RESET = "\033[0m"

BAR_W = 38

def bar(p):
    n = int(p * BAR_W)
    return f"[{'█'*n}{'░'*(BAR_W-n)}] {p*100:.1f}%"

def colour(label):
    return f"{RED}{BOLD}{label}{RESET}" if label=="PHISHING" else f"{GREEN}{BOLD}{label}{RESET}"

def print_result(label, conf, feats, subject=""):
    phish_p = conf if label=="PHISHING" else 1-conf
    print("\n" + "─"*54)
    if subject:
        print(f"  Email    : {subject[:50]}")
    print(f"  Verdict  : {colour(label)}")
    print(f"  Risk     : {bar(phish_p)}")
    print(f"  Phishing probability : {conf*100:.2f}%")
    print("  Key signals:")
    print(f"    URLs detected      : {feats['num_urls']}")
    print(f"    IP-based URL       : {'⚠ YES' if feats['has_ip_url'] else 'No'}")
    print(f"    Urgent keywords    : {feats['num_urgent']}")
    print(f"    ALL-CAPS words     : {feats['num_caps']}")
    print(f"    Exclamation marks  : {feats['num_exclaim']}")
    print(f"    HTML tags          : {feats['num_html_tags']}")
    print(f"    Form tag found     : {'⚠ YES' if feats['has_form'] else 'No'}")
    print("─"*54 + "\n")


def do_analyse(model, text, subject=""):
    label, conf = model.predict(text)
    feats       = extract_features(text)
    print_result(label, conf, feats, subject)
    rid = save_result(subject or text[:60], text, label, conf, feats)
    print(f"  Saved to database (ID #{rid})\n")


def do_history(limit=20):
    rows = get_history(limit)
    if not rows:
        print("\n  No records yet. Run an analysis first.\n"); return
    print(f"\n  {'ID':<5} {'Timestamp':<20} {'Verdict':<12} {'Conf%':>6}  {'URLs':>4}  Subject")
    print("  " + "─"*70)
    for r in rows:
        rid, ts, subj, verdict, conf, nurls, _ = r
        marker = "⚠ " if verdict=="PHISHING" else "  "
        print(f"  {rid:<5} {str(ts)[:19]:<20} {marker}{verdict:<10} {conf*100:>6.1f}%  {nurls:>4}  {str(subj)[:28]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Analyser — Baljinder Singh")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",        help="Email text to analyse")
    group.add_argument("--file",        help="Path to a .txt email file")
    group.add_argument("--history",     action="store_true", help="Show analysis history")
    group.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--limit", type=int, default=20, help="History rows to show")
    args = parser.parse_args()

    if args.history:
        do_history(args.limit); return

    if not os.path.exists(MODEL):
        print("\n  [!] No trained model found. Run:  python train.py\n"); sys.exit(1)

    model = PhishingDetector.load()

    if args.text:
        do_analyse(model, args.text)

    elif args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        do_analyse(model, text, subject=os.path.basename(args.file))

    elif args.interactive:
        print("\n  Interactive mode — paste email text, blank line to submit, 'quit' to exit.\n")
        while True:
            print("  Enter email text:")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.lower() == "quit":
                    print("  Goodbye!\n"); return
                if line == "" and lines:
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                do_analyse(model, text)


if __name__ == "__main__":
    main()
