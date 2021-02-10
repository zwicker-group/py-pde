from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class PackageConfigDirective(SphinxDirective):
    """ directive that displays all package configuration items """
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        from pde.tools.config import Config

        c = Config()
        items = []

        for p in c.data.values():
            value = c[p.name]
            item = nodes.paragraph()
            item += nodes.strong(p.name, p.name)
            descr = f"{p.description} (Default value: {value!r})"
            item += nodes.paragraph(descr, descr)
            items += item

        return items


def setup(app):
    app.add_directive("package_configuration", PackageConfigDirective)
    return {"version": "1.0.0"}
