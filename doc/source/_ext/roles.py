from docutils import nodes


def user_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Role to link to a GitHub profile: :user:`Display Name <username>`."""
    if "<" in text and text.endswith(">"):
        display, username = text[:-1].rsplit("<", 1)
        display = display.strip()
        username = username.strip()
    else:
        display = text.strip()
        username = text.strip()
    url = f"https://github.com/{username}"
    node = nodes.reference(rawtext, display, refuri=url, **options)
    return [node], []


def setup(app):
    app.add_role("user", user_role)
    return {"parallel_read_safe": True}
