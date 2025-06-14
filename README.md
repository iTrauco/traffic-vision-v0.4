# Traffic Vision

Machine learning framework for traffic analysis and autonomous systems research.

## Project Overview

This project is still in development, but aims to explore the intersection of fourth industrial revolution megatrends (AI, autonomous vehicles) and the impact of their rapid emergence on the legacy and obsolete human-made systems of our world.

The intent behind this body of work is to produce high-quality documentation and methodologies that enable researchers and citizen scientists to leverage publicly available digital infrastructure for their own investigations. Using GDOT traffic camera feeds as the primary data source, this framework captures methods and techniques that practitioners can adapt as an applied practice across any number of scientific domains. It documents various aspects of the technology stack from systems-level implementations and scripts to storage solutions and ML methodologies, providing reusable patterns that extend beyond traffic analysis alone.

## Project Structure

```
├── environment.yml          # Dependencies
├── lib/                     # Custom libraries
│   ├── mlops/              # ML operations utilities
│   │   └── preprocessing/   # Video processing
│   └── notebook_tools/     # Jupyter utilities
│       ├── export/         # Notebook export tools
│       └── widgets/        # Interactive widgets
├── notebooks/              # Development notebooks
│   ├── MLOps/
│   ├── Systems/
│   ├── Templates/
│   └── Tests/
└── README.md
```

## Branch Strategy

- **Master**: Project core
- **Develop**: Testing and merging different workflows and infrastructure improvements
- **Case study branches**: Self-contained analysis studies

## Case Studies

Each case study is scoped to its own branch. Useful findings, analysis reviews, and workflow refinements are integrated through the case study lifecycle and merged into develop, then master.



## Development Notes

This work is iterative and future clean rebuilds are likely as the framework evolves. Previous iterations:
- `traffic-vision-v0.1` (archived)
- `traffic-vision-v0.2` (archived) 
- `traffic-vision-v0.3` (archived)
- `traffic-vision-v0.4` (current v0.4.0)

Moving forward, major rebuilds will use semantic versioning within this repository. Upon completion of development and clean rebuilds, the final project will be transitoned to a permanent repository nameD `traffic-vision`.
---
**Author:** Christopher Trauco | [ORCID: 0009-0005-8113-6528](https://orcid.org/0009-0005-8113-6528)