# Ansible Deployment for Multimodal Stack

This Ansible playbook securely deploys JWT_SECRET and HF_TOKEN using Ansible Vault.

## Prerequisites

1. Install Ansible:

   ```bash
   pip install ansible
   ```

2. Create vault password file:
   ```bash
   echo "your-vault-password" > .vault_pass
   chmod 600 .vault_pass
   ```

## Setup

1. Create encrypted variables:

   ```bash
   # Create new vault file
   ansible-vault create group_vars/all/vault.yml --vault-password-file .vault_pass

   # Add these variables:
   vault_hf_token: "your-actual-huggingface-token"
   vault_jwt_secret: "your-super-secret-jwt-key"
   ```

2. Edit existing vault:
   ```bash
   ansible-vault edit group_vars/all/vault.yml --vault-password-file .vault_pass
   ```

## Deployment

Deploy to different environments:

```bash
# Deploy to development
ansible-playbook -i inventory/hosts.ini secure-gateway.yml \
  --vault-password-file .vault_pass \
  --limit dev

# Deploy to staging
ansible-playbook -i inventory/hosts.ini secure-gateway.yml \
  --vault-password-file .vault_pass \
  --limit staging

# Deploy to production
ansible-playbook -i inventory/hosts.ini secure-gateway.yml \
  --vault-password-file .vault_pass \
  --limit prod
```

## Using with prompt

Run with vault password prompt:

```bash
ansible-playbook -i inventory/hosts.ini secure-gateway.yml --ask-vault-pass
```

## Security Best Practices

1. **Never commit `.vault_pass` to git** - add it to `.gitignore`
2. Store vault password in a secure password manager
3. Use different JWT secrets for each environment
4. Rotate secrets regularly
5. Limit access to production vault files

## Viewing Encrypted Variables

To view encrypted variables:

```bash
ansible-vault view group_vars/all/vault.yml --vault-password-file .vault_pass
```

## Directory Structure

```
ansible/
├── secure-gateway.yml      # Main playbook
├── inventory/
│   └── hosts.ini          # Server inventory
├── group_vars/
│   ├── all/
│   │   └── vault.yml      # Encrypted secrets (all environments)
│   ├── dev/
│   │   └── vars.yml       # Dev-specific variables
│   ├── staging/
│   │   └── vars.yml       # Staging-specific variables
│   └── prod/
│       └── vars.yml       # Prod-specific variables
└── templates/
    └── env.j2             # .env file template
```
