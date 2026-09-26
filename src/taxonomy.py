"""The issue taxonomy: what a q-commerce customer can complain about (or praise).

Designed MECE-style around the customer journey: browse -> pay -> fulfil ->
deliver -> after-sales. Each topic maps to an owning function, so every
finding lands on someone's desk.
"""

TOPICS = {
    "delivery_speed": ("Late or slow delivery, missed the promised ETA, order took too long", "Last-mile ops"),
    "missing_wrong_items": ("Items missing, wrong item delivered, damaged or leaking package", "Dark-store ops"),
    "product_quality": ("Expired, stale, rotten or poor-quality products, especially fruit, veg, dairy", "Category / supply"),
    "availability": ("Items out of stock, limited range, store unavailable or closed in their area", "Category / supply"),
    "pricing_fees": ("High prices vs. local stores or rivals, handling / platform / surge / small-cart fees", "Pricing"),
    "offers_coupons": ("Coupons not applying, misleading offers, cashback not credited, membership value", "Marketing"),
    "payment": ("Payment failed, money debited but order not placed, COD or UPI problems", "Payments / tech"),
    "refund_returns": ("Refund delayed or denied, return or replacement not processed", "Customer support"),
    "customer_support": ("Support unreachable, bot-only chat, unhelpful or rude agents, issue unresolved", "Customer support"),
    "delivery_partner": ("Rider behaviour: rude, won't come to door, asks for extra cash, fake 'delivered'", "Last-mile ops"),
    "app_experience": ("Crashes, bugs, login/OTP issues, slow app, confusing UI, too many ads or notifications", "Product / tech"),
    "account_issues": ("Account blocked, suspected fraud, address or location problems", "Product / tech"),
}

TOPIC_LABELS = {
    "delivery_speed": "Delivery speed",
    "missing_wrong_items": "Missing / wrong items",
    "product_quality": "Product quality",
    "availability": "Availability",
    "pricing_fees": "Pricing & fees",
    "offers_coupons": "Offers & coupons",
    "payment": "Payments",
    "refund_returns": "Refunds & returns",
    "customer_support": "Customer support",
    "delivery_partner": "Delivery partner",
    "app_experience": "App experience",
    "account_issues": "Account issues",
}

SENTIMENTS = ["positive", "neutral", "negative"]
COMPETITORS = ["Blinkit", "Zepto", "Instamart", "bigbasket", "Flipkart Minutes", "Amazon", "JioMart", "DMart",
               "Local store", "Other", "None"]


def taxonomy_prompt() -> str:
    lines = [f"- {k}: {desc}" for k, (desc, _) in TOPICS.items()]
    return "\n".join(lines)
