def detect_suspicious(log_entry):
    keywords = ['login', 'admin', 'hack', 'password']
    return any(kw in log_entry['url'].lower() for kw in keywords)
