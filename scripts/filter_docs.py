import re
from collections import defaultdict

def is_gibberish(text, min_word_ratio=0.4, min_words=30, min_avg_length=4, min_vowel_ratio=0.38):
    tokens = text.split()
    if len(tokens) < min_words:
        return True
    alpha_words = [t for t in tokens if re.match(r'^[^\W\d_]{3,}$', t, re.UNICODE)]
    if len(alpha_words) / len(tokens) < min_word_ratio:
        return True
    avg_length = sum(len(t) for t in alpha_words) / len(alpha_words)
    if avg_length < min_avg_length:
        return True
    all_alpha_chars = "".join(alpha_words).lower()
    vowel_ratio = sum(1 for c in all_alpha_chars if c in "aeiouáéíóúü") / len(all_alpha_chars)
    return vowel_ratio < min_vowel_ratio

def remove_gibberish(data, min_word_ratio):
    # Clean pages: remove gibberish
    file_content = []
    for page, content in enumerate(data):
        content = content[1] if isinstance(content, tuple) else content
        cleaned = "" if is_gibberish(content, min_word_ratio) else content
        file_content.append(cleaned)
    
    return file_content

def empty(content_list, alpha_ratio):
    mask = [1 for x in content_list if len(x) != 0]
    return sum(mask) / len(content_list) < (1 - alpha_ratio)

def gibberish_docs(docs, alpha_ratio, min_word_ratio):
    gibberish = []
    for file, data in docs.items():
        # Don't include files with only 1 page (usually cover pages)
        if len(data) <= 1:
            continue
        
        # Replace gibberish pages with empty string
        file_content = remove_gibberish(data, min_word_ratio)
        
        # Don't include files with less than 30% non-gibberish content
        if empty(file_content, alpha_ratio):
            gibberish.append(file)
    
    return gibberish

def filter_documents(docs, alpha_ratio, min_word_ratio):
    documents = defaultdict(list)

    for file, data in docs.items():
        # Don't include files with only 1 page (usually cover pages)
        if len(data) <= 1:
            continue
        
        # Clean pages: remove gibberish
        file_content = []
        for page, content in enumerate(data):
            content = content[1] if isinstance(content, tuple) else content
            cleaned = "" if is_gibberish(content, min_word_ratio) else content
            # file_content.append(cleaned)
            documents[file].append((page + 1, cleaned))

    return documents

def inject_new_paths(file_list, new_root_files, old_root, new_root):
    return [new_root / p.relative_to(old_root) if p.name in new_root_files else p for p in file_list]