# Developer guide

This section covers the package architecture, testing, validation and model
extensions. For an analysis, use the [user guide](tutorials/tirt_workflow.ipynb).
Supported environments are described under [installation](installation.md);
model assumptions and response formats are in the [model reference](models/index.md).

```{toctree}
:maxdepth: 1

architecture
requirements
testing
recovery
tutorial_migration
roadmap
models/extensions
models/statistical_audit
models/thurstonian_roadmap
models/tirt_rank
```


## Documentation website

To run the complete tutorial notebooks, including their `%watermark -iv`
version stamps, install the notebook dependencies from a local checkout:

```console
python -m pip install ".[docs,plots,io,covariates]"
```

Use a notebook kernel from that environment. Ordinary installation and the
Quick start do not require the documentation-building tools.

The website uses the existing Sphinx sources in `docs/`, including saved
notebook outputs. Documentation builds do not execute notebooks or fit models.
Build locally with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs _build/html
```

Pull requests affecting the documentation run the **Documentation** workflow.
Download its `documentation-html` artifact and open `index.html` to preview the
website. Pull requests do not publish to GitHub Pages.

In repository **Settings → Pages**, select **GitHub Actions** as the source once.
Every push or merge to `main` then builds and publishes the documentation
automatically. A failed build prevents publication. Only built HTML is uploaded.
The project address is `https://qipengchen.github.io/pyfcirt/`.

To republish manually, open **Actions → Documentation → Run workflow**, select
`main`, and enable **Publish the main branch documentation to GitHub Pages**.
A manual run without that option only builds a preview. Other branches and pull
requests do not publish.
