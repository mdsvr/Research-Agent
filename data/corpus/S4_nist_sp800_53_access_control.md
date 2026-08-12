# NIST Special Publication 800-53 Revision 5: Access Control (AC) Security Controls

## AC-2 Account Management
The organization manages information system accounts, including establishing, activating, modifying, reviewing, disabling, and removing accounts in accordance with organizational procedures.

### Control Enhancements
- **AC-2(1) Automated System Account Management:** Employs automated mechanisms to support the management of information system accounts.
- **AC-2(3) Disable Inactive Accounts:** Automatically disables inactive accounts after 90 days of operational inactivity.

## AC-3 Access Enforcement
The information system enforces approved authorizations for logical access to information and system resources in accordance with applicable access control policies (e.g., identity-based, role-based access control RBAC, mandatory access control MAC).

## AC-7 Unsuccessful Logon Attempts
The information system automatically locks the account until released by an administrator when the maximum number of unsuccessful logon attempts (5 consecutive invalid attempts) is exceeded within a 15-minute window.

## AC-17 Remote Access
Enforces explicit authorization for remote access connections. Employs cryptographic mechanisms (TLS 1.3 or IPsec VPNs) to protect the confidentiality and integrity of remote access sessions, mandatory multi-factor authentication (MFA) for all remote management access.
