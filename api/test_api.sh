#!/bin/bash
# Test script for the Rust Sentiment API

API_URL="http://localhost:3000"

echo "=========================================="
echo "Testing Rust Sentiment Analysis API"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4

    echo -e "${BLUE}Test: ${description}${NC}"
    echo "Endpoint: ${method} ${endpoint}"

    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" "${API_URL}${endpoint}")
    else
        response=$(curl -s -w "\n%{http_code}" -X "${method}" "${API_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "${data}")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✓ Success (HTTP ${http_code})${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo -e "${RED}✗ Failed (HTTP ${http_code})${NC}"
        echo "$body"
    fi

    echo ""
}

# Test 1: Health check
test_endpoint "GET" "/health" "" "Health Check"

# Test 2: Single positive review
test_endpoint "POST" "/predict" \
    '{"text": "This restaurant is amazing! Best food ever!"}' \
    "Single Prediction - Positive Review"

# Test 3: Single negative review
test_endpoint "POST" "/predict" \
    '{"text": "Terrible service and awful food. Never coming back."}' \
    "Single Prediction - Negative Review"

# Test 4: Neutral review
test_endpoint "POST" "/predict" \
    '{"text": "It was okay, nothing special."}' \
    "Single Prediction - Neutral Review"

# Test 5: Batch predictions with Yelp reviews
test_endpoint "POST" "/batch" \
    '{
        "texts": [
            "Wow... Loved this place.",
            "Crust is not good.",
            "The fries were great too.",
            "Would not go back.",
            "Service was very prompt.",
            "Highly recommended.",
            "Not impressed.",
            "The food here is mediocre at best."
        ]
    }' \
    "Batch Predictions - Yelp Reviews"

# Test 6: Empty text (should fail)
echo -e "${BLUE}Test: Error Handling - Empty Text${NC}"
echo "Endpoint: POST /predict"
response=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": ""}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" -eq 400 ]; then
    echo -e "${GREEN}✓ Correctly rejected empty text (HTTP ${http_code})${NC}"
    echo "$body"
else
    echo -e "${RED}✗ Should have rejected empty text${NC}"
    echo "$body"
fi
echo ""

# Test 7: Large batch (should work up to 100)
echo -e "${BLUE}Test: Batch Predictions - 10 Reviews${NC}"
echo "Endpoint: POST /batch"

texts='['
for i in {1..10}; do
    texts+="\"Review number $i: This is a test review.\""
    if [ $i -lt 10 ]; then
        texts+=","
    fi
done
texts+=']'

response=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/batch" \
    -H "Content-Type: application/json" \
    -d "{\"texts\": $texts}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" -eq 200 ]; then
    result_count=$(echo "$body" | jq '.results | length')
    echo -e "${GREEN}✓ Batch processed successfully (${result_count} results)${NC}"
else
    echo -e "${RED}✗ Batch processing failed${NC}"
    echo "$body"
fi
echo ""

echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
