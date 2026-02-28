#!/bin/bash

# init-letsencrypt.sh - Initialize Let's Encrypt SSL certificates
# Run this script ONCE on first deployment

set -e

DOMAIN="lightrag.windyselfhost.space"
EMAIL="your-email@example.com"  # TODO: Change to your email
STAGING=0  # Set to 1 to test with Let's Encrypt staging (no rate limits)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "### Starting SSL certificate initialization for $DOMAIN ..."

# Check if docker compose is available
if docker compose version > /dev/null 2>&1; then
    COMPOSE="docker compose"
elif docker-compose version > /dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "Error: docker compose not found!"
    exit 1
fi

# Create directories
mkdir -p ./certbot-www
mkdir -p ./certbot-conf

# Create dummy certificate so nginx can start
echo "### Creating dummy certificate for $DOMAIN ..."
CERT_PATH="./certbot-conf/live/$DOMAIN"
mkdir -p "$CERT_PATH"

# Generate self-signed cert
openssl req -x509 -nodes -newkey rsa:4096 -days 1 \
    -keyout "$CERT_PATH/privkey.pem" \
    -out "$CERT_PATH/fullchain.pem" \
    -subj "/CN=$DOMAIN" 2>/dev/null

echo "### Starting nginx with dummy certificate ..."
$COMPOSE up -d nginx

echo "### Waiting for nginx to start ..."
sleep 5

echo "### Removing dummy certificate ..."
rm -rf "$CERT_PATH"

echo "### Requesting Let's Encrypt certificate for $DOMAIN ..."

# Select staging or production
if [ $STAGING != "0" ]; then
    STAGING_ARG="--staging"
    echo "### WARNING: Using Let's Encrypt STAGING environment (certificates will NOT be trusted)"
else
    STAGING_ARG=""
fi

$COMPOSE run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    $STAGING_ARG \
    -d "$DOMAIN"

echo "### Reloading nginx with real certificate ..."
$COMPOSE exec nginx nginx -s reload

echo ""
echo "### SSL certificate initialized successfully!"
echo "### Your site should now be accessible at: https://$DOMAIN"
echo ""
echo "### To start all services:"
echo "    cd $(pwd) && $COMPOSE up -d"
echo ""
echo "### Certificate auto-renewal is handled by the certbot container."
