# OWASP Top 10 Security Focus: Injection and Broken Access Control

## A03:2021 – Injection
Injection flaws, such as SQL, NoSQL, OS Command, and LDAP injection, occur when untrusted data is sent to an interpreter as part of a command or query. The attacker's hostile data can trick the interpreter into executing unintended commands or accessing data without proper authorization.

### Common Vectors
- **SQL Injection (SQLi):** Parameter substitution in dynamic string concatenation allows arbitrary SQL commands to execute against relational databases (e.g., PostgreSQL, MySQL).
- **Command Injection:** User input passed directly to shell execution functions (`exec()`, `system()`, `popen()`) without sanitization allows execution of arbitrary operating system commands.

### Prevention Controls
1. Use parameterized queries (also known as prepared statements) for all database access.
2. Use positive server-side input validation with strict allowlists.
3. Escape special characters using context-aware escaping APIs when parameterized interfaces are unavailable.

## A01:2021 – Broken Access Control
Access control enforces policy such that users cannot act outside of their intended permissions. Failures typically lead to unauthorized information disclosure, modification, or destruction of all data or performing a business function outside the user's limits.

### Prevention Controls
- Enforce least privilege by default; deny all access unless explicitly permitted.
- Implement access control mechanisms once and re-use them throughout the application, including minimizing CORS usage.
- Disable web server directory listing and ensure file metadata is not accessible within web roots.
