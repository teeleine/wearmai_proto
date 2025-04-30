import re

def remove_css_styles(input_string: str) -> str:
    # Step 1: Remove CSS blocks (content inside { })
    cleaned_string = re.sub(r'\{[^}]*\}', '', input_string)

    # Step 2: Define a list of common CSS selectors to remove
    selectors = [
        'body', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'div', 'span', 'figure', 'figcaption',
        'section', 'article', 'header', 'footer',
        'nav', 'aside', 'table', 'th', 'tr', 'td',
        'ul', 'ol', 'li', 'a', 'img', 'button',
        'input', 'textarea', 'form', 'label'
    ]

    # Step 3: Build a regular expression to match and remove these selectors
    selector_pattern = r'\b(?:' + '|'.join(selectors) + r')\b'
    cleaned_string = re.sub(selector_pattern, '', cleaned_string)

    # Step 4: Clean up extra spaces but preserve line breaks
    cleaned_string = re.sub(r'[ \t]+', ' ', cleaned_string)  # Normalize spaces
    cleaned_string = re.sub(r'\n\s*\n', '\n\n', cleaned_string)  # Preserve paragraph breaks

    return cleaned_string.strip()

def remove_figure_references(input_string: str) -> str:
    # Regex to match lines that start with 'Figure', followed by a number, a colon, and any word after it
    figure_pattern = r'^Figure\s+\d+:\s+\w+.*$'
    
    # Remove matching lines
    cleaned_string = re.sub(figure_pattern, '', input_string, flags=re.MULTILINE)
    
    # Clean up any extra newlines left behind
    cleaned_string = re.sub(r'\n\s*\n', '\n\n', cleaned_string).strip()
    
    return cleaned_string


def clean_knowledge_base(input_book: str) -> str:
    clean_book = remove_css_styles(input_book).replace(".", "").replace(",", "")
    clean_book = remove_figure_references(clean_book)

    return clean_book