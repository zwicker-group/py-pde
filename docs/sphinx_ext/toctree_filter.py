import re

from sphinx.directives.other import TocTree


class TocTreeFilter(TocTree):
    """directive to filter table-of-contents entries """

    hasPat = re.compile("^\s*:(.+):(.+)$")

    # Remove any entries in the content that we dont want and strip
    # out any filter prefixes that we want but obviously don't want the
    # prefix to mess up the file name.
    def filter_entries(self, entries):
        excl = self.state.document.settings.env.config.toc_filter_exclude
        filtered = []
        for e in entries:
            m = self.hasPat.match(e)
            if m != None:
                if not m.groups()[0] in excl:
                    filtered.append(m.groups()[1])
            else:
                filtered.append(e)
        return filtered

    def run(self):
        # Remove all TOC entries that should not be on display
        self.content = self.filter_entries(self.content)
        return super().run()


def setup(app):
    app.add_config_value("toc_filter_exclude", [], "html")
    app.add_directive("toctree-filt", TocTreeFilter)
    return {"version": "1.0.0"}
