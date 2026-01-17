# OpenAD-lib Development Plan

## 📋 Executive Summary

This comprehensive plan identifies gaps, inconsistencies, and missing features in the OpenAD-lib library based on analysis of the current codebase, documentation, and identified issues during README validation.

---

## 🎯 Current State Assessment

### ✅ What's Working Well

1. **Core Models Implemented**
   - ✅ ADM1 (38-state mechanistic model)
   - ✅ AM2 (4-state simplified model)
   - ✅ LSTM (time series ML model)
   - ✅ Multi-Task GP (uncertainty-aware ML model)

2. **Control & Optimization**
   - ✅ AM2MPC (do-mpc based controller)
   - ✅ AM2Calibrator (Optuna-based)
   - ✅ ADM1Calibrator (Optuna-based)

3. **Data & Preprocessing**
   - ✅ Sample datasets (6 datasets)
   - ✅ ACoD preprocessing module
   - ✅ Feedstock library (12 substrates)
   - ✅ Data validation utilities

4. **Visualization**
   - ✅ Unified plotting system
   - ✅ Consistent styling across all plots
   - ✅ Publication-ready figures

### ❌ Critical Gaps Identified

1. **No Test Suite**
   - Zero test files found
   - No CI/CD pipeline
   - No automated quality checks

2. **Incomplete Documentation**
   - No API reference documentation
   - No MkDocs setup (listed in dependencies but not configured)
   - Missing developer guide
   - No contribution guidelines

3. **API Inconsistencies** (Fixed during README validation)
   - Method naming (`update_parameters` vs `update_params`)
   - Return type inconsistencies (`simulate()` vs `run()`)
   - Column name standardization issues

4. **Missing Features** (Mentioned but not implemented)
   - Bayesian optimization beyond Optuna
   - Advanced scheduling algorithms
   - Real-time data integration
   - Model comparison framework

---

## 🚨 Priority 1: Critical Infrastructure (2-3 weeks)

### 1.1 Testing Framework

**Goal**: Achieve 80%+ code coverage

#### Tasks:
- [ ] **Set up pytest infrastructure** (2 days)
  - Create `tests/` directory structure
  - Configure `pytest.ini` with coverage settings
  - Set up GitHub Actions for CI/CD

- [ ] **Unit Tests** (1 week)
  - `tests/test_models/`
    - `test_adm1_model.py` - Core ADM1 functionality
    - `test_am2_model.py` - Core AM2 functionality
    - `test_lstm_model.py` - LSTM basics
    - `test_mtgp.py` - Multi-Task GP
  - `tests/test_data/`
    - `test_loaders.py` - Data loading functions
    - `test_datasets.py` - BiogasDataset, FeedstockDataset
    - `test_validators.py` - Validation functions
  - `tests/test_utils/`
    - `test_metrics.py` - Metric calculations
    - `test_plots.py` - Plotting functions (visual regression)

- [ ] **Integration Tests** (3 days)
  - End-to-end workflow tests
  - Calibration pipeline tests
  - MPC control loop tests

- [ ] **Example Validation Tests** (2 days)
  - Automated tests that run all README examples
  - Notebook execution tests
  - Output validation

#### Deliverables:
```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_models/
│   ├── test_adm1_model.py
│   ├── test_am2_model.py
│   ├── test_lstm_model.py
│   └── test_mtgp.py
├── test_data/
│   ├── test_loaders.py
│   ├── test_datasets.py
│   └── test_validators.py
├── test_utils/
│   ├── test_metrics.py
│   └── test_plots.py
├── test_optimisation/
│   ├── test_am2_calibration.py
│   └── test_adm1_calibration.py
├── test_control/
│   └── test_am2_mpc.py
└── test_integration/
    ├── test_workflows.py
    └── test_examples.py
```

### 1.2 CI/CD Pipeline

- [ ] **GitHub Actions Workflow** (1 day)
  ```yaml
  # .github/workflows/tests.yml
  name: Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: [3.8, 3.9, '3.10', 3.11]
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ matrix.python-version }}
        - name: Install dependencies
          run: pip install -e .[dev]
        - name: Run tests
          run: pytest --cov=openad_lib --cov-report=xml
        - name: Upload coverage
          uses: codecov/codecov-action@v3
  ```

- [ ] **Pre-commit Hooks** (1 day)
  - Black formatting
  - isort import sorting
  - Flake8 linting
  - Type checking with mypy

### 1.3 Documentation Infrastructure

- [ ] **Set up MkDocs** (2 days)
  ```
  docs/
  ├── index.md
  ├── getting-started/
  │   ├── installation.md
  │   └── quickstart.md
  ├── user-guide/
  │   ├── models.md
  │   ├── calibration.md
  │   ├── control.md
  │   └── visualization.md
  ├── api-reference/
  │   ├── models.md
  │   ├── data.md
  │   ├── utils.md
  │   └── control.md
  ├── developer-guide/
  │   ├── contributing.md
  │   ├── architecture.md
  │   └── testing.md
  └── examples/
      ├── adm1-tutorial.md
      ├── am2-calibration.md
      └── mtgp-uncertainty.md
  ```

- [ ] **Auto-generate API docs** (1 day)
  - Use mkdocstrings for docstring parsing
  - Generate from code annotations

- [ ] **Deploy to GitHub Pages** (0.5 day)

---

## 🔧 Priority 2: API Consistency & Refactoring (1-2 weeks)

### 2.1 Standardize Return Types

**Issue**: `simulate()` returns dict, `run()` returns DataFrame

**Solution**:
- [ ] **Create unified return type** (2 days)
  ```python
  @dataclass
  class SimulationResult:
      """Unified simulation result container."""
      data: pd.DataFrame
      metadata: Dict[str, Any]
      
      def to_dict(self) -> Dict[str, np.ndarray]:
          """Convert to dict format."""
          
      def to_dataframe(self) -> pd.DataFrame:
          """Get DataFrame representation."""
  ```

- [ ] **Update all models to use SimulationResult** (3 days)
  - AM2Model
  - ADM1Model
  - Base class

### 2.2 Standardize Method Names

**Issues Found**:
- `update_params()` vs `update_parameters()`
- `load_data_from_dataframe()` has inconsistent parameter names

**Solution**:
- [ ] **Audit all public methods** (1 day)
- [ ] **Create naming convention guide** (0.5 day)
- [ ] **Refactor with deprecation warnings** (2 days)
  ```python
  def update_parameters(self, params: Dict):
      """Deprecated: Use update_params() instead."""
      warnings.warn("update_parameters is deprecated, use update_params", DeprecationWarning)
      return self.update_params(params)
  ```

### 2.3 Standardize DataFrame Columns

**Issue**: AM2 creates standardized columns (`S1out`, `S2out`, `Q`) but this isn't documented

**Solution**:
- [ ] **Document column name conventions** (1 day)
  - Create `docs/conventions.md`
  - List all standardized column names for each model

- [ ] **Add column name constants** (1 day)
  ```python
  # src/openad_lib/constants.py
  class AM2Columns:
      """Standard column names for AM2 model."""
      S1_OUT = 'S1out'
      S2_OUT = 'S2out'
      Q = 'Q'
      # ...
  ```

---

## 🆕 Priority 3: New Features (2-4 weeks)

### 3.1 Model Comparison Framework

**Goal**: Easy comparison between models

- [ ] **Create ModelComparison class** (3 days)
  ```python
  class ModelComparison:
      """Compare multiple models on same dataset."""
      def __init__(self, models: List[BaseModel]):
          self.models = models
      
      def compare(self, data) -> pd.DataFrame:
          """Run all models and compare metrics."""
      
      def plot_comparison(self):
          """Visualize model predictions side-by-side."""
  ```

- [ ] **Add benchmark suite** (2 days)
  - Standard benchmark datasets
  - Automated scoring

### 3.2 Enhanced Uncertainty Quantification

- [ ] **Ensemble predictions** (3 days)
  - Bootstrap ensembles for mechanistic models
  - Multiple MTGP models

- [ ] **Uncertainty visualization** (2 days)
  - Confidence band plots
  - Prediction interval coverage plots

### 3.3 Real-time Data Integration

- [ ] **Data streaming support** (1 week)
  ```python
  class DataStream:
      """Handle real-time sensor data."""
      def connect(self, source: str):
          """Connect to data source."""
      
      def get_latest(self, n_points: int = 100) -> pd.DataFrame:
          """Get latest n data points."""
  ```

- [ ] **Online calibration** (3 days)
  - Incremental model updates
  - Sliding window optimization

### 3.4 Advanced Scheduling

- [ ] **Multi-objective optimization** (1 week)
  - Biogas production maximization
  - Cost minimization
  - Stability constraints

- [ ] **Genetic algorithm scheduler** (3 days)

---

## 📊 Priority 4: Data & Examples (1 week)

### 4.1 More Sample Datasets

- [ ] **Add industrial-scale data** (2 days)
  - WWTP data
  - Agricultural biogas plant data

- [ ] **Add co-digestion examples** (1 day)
  - Multiple feedstock combinations

### 4.2 More Example Scripts

- [ ] **Industry use cases** (3 days)
  - `examples/08_industrial_scale.py`
  - `examples/09_acod_optimization.py`
  - `examples/10_uncertainty_analysis.py`

---

## 🐛 Priority 5: Bug Fixes & Technical Debt (Ongoing)

### 5.1 Known Issues

- [ ] **BiogasDataset needs from_csv() class method** (0.5 day)
  ```python
  @classmethod
  def from_csv(cls, path: str, **kwargs) -> 'BiogasDataset':
      """Load dataset from CSV file."""
      data = pd.read_csv(path)
      return cls(data, **kwargs)
  ```

- [ ] **Add type hints throughout** (1 week)
  - All public methods
  - All internal functions
  - Run mypy validation

- [ ] **Improve error messages** (ongoing)
  - Add context to ValueError messages
  - Suggest solutions

### 5.2 Performance Optimization

- [ ] **Profile simulation performance** (2 days)
  - Identify bottlenecks
  - Optimize ODE solvers

- [ ] **Add caching** (1 day)
  - Cache expensive computations
  - Memoize parameter validation

---

## 📈 Success Metrics

### Testing
- [ ] 80%+ code coverage
- [ ] All examples pass automated tests
- [ ] CI/CD passing on all Python versions (3.8-3.11)

### Documentation
- [ ] Complete API reference
- [ ] 5+ comprehensive tutorials
- [ ] Developer guide published

### Quality
- [ ] Zero critical bugs
- [ ] All deprecation warnings resolved
- [ ] Type hints coverage >90%

### Community
- [ ] 50+ GitHub stars
- [ ] 5+ external contributors
- [ ] 10+ community examples

---

## 🗓️ Release Roadmap

### v0.2.0 (Current) - Documentation & Stability
- ✅ README examples fixed
- ✅ Unified plotting system
- ✅ API consistency fixes
- 🔄 Complete test suite
- 🔄 MkDocs documentation

### v0.3.0 (2 months) - Testing & Quality
- Complete test coverage
- CI/CD pipeline
- Type hints
- Performance optimizations

### v0.4.0 (4 months) - Advanced Features
- Model comparison framework
- Real-time data integration
- Advanced scheduling
- Enhanced uncertainty quantification

### v1.0.0 (6 months) - Production Ready
- Stable API
- Complete documentation
- Industry validation
- Long-term support commitment

---

## 🎯 Immediate Next Steps (This Week)

1. **Set up test infrastructure** (Day 1)
   - Create tests/ directory
   - Configure pytest
   - Write first model test

2. **Create CONTRIBUTING.md** (Day 1)
   - Development setup
   - Coding standards
   - PR process

3. **Set up GitHub Actions** (Day 2)
   - Basic CI workflow
   - Test automation

4. **Write core model tests** (Days 3-5)
   - AM2Model tests
   - ADM1Model tests
   - Data loader tests

---

## 📝 Notes

### Dependencies to Add
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",  # Parallel testing
    "mypy>=1.0",
    "pre-commit>=3.0",
    "mkdocstrings[python]>=0.20",
]
```

### Files to Create
- `tests/__init__.py`
- `tests/conftest.py`
- `.github/workflows/tests.yml`
- `.pre-commit-config.yaml`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `mkdocs.yml`
- `docs/index.md`

---

## ✅ Conclusion

This plan provides a clear path from the current state (functional but untested) to a production-ready library with:
- Comprehensive testing
- Complete documentation
- Consistent API
- Advanced features
- Active community

**Estimated total effort**: 8-12 weeks for v0.3.0 release
**Priority focus**: Testing & documentation (Weeks 1-4)
