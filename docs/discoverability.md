# Discoverability: search and generative engines

Reviewed 2026-11-04. This document records the evidence behind the repository's SEO and GEO (generative engine optimization) choices. These are visibility improvements, not ranking or citation guarantees.

## What the sources support

[GitHub's topic guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) says topics help people find related projects. [GitHub repository search](https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories) can match names, descriptions, topics, and READMEs. Use a precise description and a focused set of topics that reflect implemented capabilities.

[GitHub's README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) recommends explaining purpose, usefulness, getting started, help, and maintainers. Keep the repository's entry point short enough to navigate and move detailed contracts into linked documentation.

[Google's AI search guidance](https://developers.google.com/search/docs/appearance/ai-features) states that existing SEO best practices apply to AI features; special optimization is not required. Public, helpful content with accessible text and supporting evidence is a sound foundation. This source covers Google's AI features and should not be generalized into guaranteed behavior for every AI engine.

## Applied to MLSurgie

- Canonical identity: **MLSurgie**, a Python framework for **LLM-driven weight-space programming** of small neural networks.
- Relevant terms in natural prose: direct weight construction, neural network weights, tabular binary classification, PyTorch, Ollama, evaluation, and replay.
- A concise GitHub About description and focused repository topics.
- A clear README definition, capability table, runnable quickstart, explicit prototype status, and maintained help links.
- A text-accessible SVG workflow, with a matching prose explanation so the image is not required to understand the method.
- FAQ answers that state supported behavior directly and distinguish offline fixtures from interactive LLM evidence.
- A linked experiment with dataset provenance, split sizes, baselines, selection protocol, and limitations.
- Package repository/documentation URLs for visitors discovering the project through Python metadata.

No ranking guarantees, invented adoption counts, benchmark generalization claims, hidden keywords, or special AI indexing files are used. GitHub controls the repository page's HTML metadata and crawling behavior; this change does not add website-level structured data or claim control over search indexing.

## Maintenance

Keep README capabilities, package metadata, repository description, and examples consistent with releases. Add benchmark results only with methods and limitations. Periodically check README links and execute the quickstart. Use actual GitHub traffic/referrals, issue questions, and onboarding failures to guide improvements. Search results and AI citations are external outcomes that must be observed over time.
