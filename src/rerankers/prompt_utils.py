"""Prompt utilities for LLM reranking."""


def build_rerank_prompt(user_id, candidate_item, user_history, base_score, item_info=None):
    """Build prompt for reranking.

    Args:
        user_id: User ID
        candidate_item: Candidate item ID
        user_history: List of user's historical item IDs
        base_score: Base recommendation score
        item_info: Optional dict with item details

    Returns:
        Prompt string
    """
    history_str = ", ".join(user_history[:10]) if user_history else "none"

    prompt = f"""You are a recommendation assistant. Given a user's purchase history and a candidate item, rate how likely the user would be interested in the candidate item on a scale of 0-1.

User ID: {user_id}
User's purchase history: {history_str}
Candidate item: {candidate_item}
Base recommendation score: {base_score:.4f}

{item_info or ''}

Please respond with only a number between 0 and 1 representing your rating.
Rating: """

    return prompt


def parse_rerank_response(response):
    """Parse LLM response to get score.

    Args:
        response: LLM response string

    Returns:
        Score float
    """
    try:
        # Try to extract number from response
        lines = response.strip().split('\n')
        for line in reversed(lines):
            # Look for number
            import re
            numbers = re.findall(r'0\.\d+|0|1\.0', line)
            if numbers:
                return float(numbers[0])

        # Fallback: try to find any number
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[-1])
            # Normalize to 0-1
            if score > 1:
                score = score / 10.0 if score <= 10 else 0.5
            return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"Error parsing response: {e}")

    return 0.5  # Default score


def format_item_for_prompt(item_id, item_data):
    """Format item data for prompt.

    Args:
        item_id: Item ID
        item_data: Dict with item information

    Returns:
        Formatted string
    """
    if not item_data:
        return f"Item: {item_id}"

    parts = [f"Item: {item_id}"]

    if 'title' in item_data:
        parts.append(f"Title: {item_data['title']}")

    if 'description' in item_data:
        desc = item_data['description']
        if len(desc) > 100:
            desc = desc[:100] + "..."
        parts.append(f"Description: {desc}")

    if 'category' in item_data:
        parts.append(f"Category: {item_data['category']}")

    return "\n".join(parts)
