# Traffic Vision

Machine learning framework for traffic analysis and autonomous systems research.
## Project Overview

This project is still in development, but aims to explore the intersection of fourth industrial revolution megatrends (AI, autonomous vehicles, 3D printing) and the impact of their rapid emergence on the legacy and obsolete human-made systems of our world.

The primary output is documenting methods and experiments for accessing GDOT traffic camera feeds, creating reusable frameworks and methodologies that researchers and citizen scientists can leverage for their own AI/ML investigations.

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
- `traffic-vision` (current v0.4.0, permanent name)

Moving forward, major rebuilds will use semantic versioning within this repository rather than creating new repositories.

---
**Author:** Christopher Trauco | [ORCID: 0009-0005-8113-6528](https://orcid.org/0009-0005-8113-6528)