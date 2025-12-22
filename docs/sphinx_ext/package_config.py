from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class PackageConfigDirective(SphinxDirective):
    """Directive that displays all package configuration items."""

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        from pde import config

        items = []

        for name, p in config.to_dict().items():
            description = nodes.paragraph(text=p.description + " ")
            description += nodes.strong(text=f"(Default value: {config[name]!r})")

            items += nodes.definition_list_item(
                "",
                nodes.term(text=name),
                nodes.definition("", description),
            )

        return [nodes.definition_list("", *items)]


def setup(app):
    app.add_directive("package_configuration", PackageConfigDirective)
    return {"version": "1.0.0"}
