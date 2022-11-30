from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class PackageConfigDirective(SphinxDirective):
    """directive that displays all package configuration items"""

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        from pde.tools.config import Config

        c = Config()
        items = []

        for p in c.data.values():
            description = nodes.paragraph(text=p.description + " ")
            description += nodes.strong(text=f"(Default value: {c[p.name]!r})")

            items += nodes.definition_list_item(
                "",
                nodes.term(text=p.name),
                nodes.definition("", description),
            )

        return [nodes.definition_list("", *items)]


def setup(app):
    app.add_directive("package_configuration", PackageConfigDirective)
    return {"version": "1.0.0"}
