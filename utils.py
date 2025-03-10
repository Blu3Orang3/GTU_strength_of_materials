def tooltip(text, concept=None):
    if concept:
        return f'<span title="{ENGINEERING_TOOLTIPS.get(concept, "")}">{text}</span>'
    return text
