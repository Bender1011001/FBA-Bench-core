import asyncio

from agent_runners.registry import create_runner
from fba_bench_core.benchmarking.metrics.registry import MetricRegistry
from fba_bench_core.benchmarking.scenarios.registry import scenario_registry
from fba_bench_core.benchmarking.validators.registry import ValidatorRegistry

from .models import EngineConfig, EngineReport, RunReport, ScenarioReport


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config

    async def run(self) -> EngineReport:
        scenario_reports = []
        for scenario_spec in self.config.scenarios:
            scenario_fn = scenario_registry.get(scenario_spec.key)
            if scenario_fn is None:
                raise ValueError(
                    f"Scenario '{scenario_spec.key}' not found in registry"
                )

            # Assume first runner for minimal implementation
            if not self.config.runners:
                raise ValueError("No runners configured")
            runner_spec = self.config.runners[0]
            runner = create_runner(runner_spec.key, runner_spec.config)

            repetitions = scenario_spec.repetitions or 1
            seeds = scenario_spec.seeds or [42] * repetitions
            scenario_runs = []
            timeout = scenario_spec.timeout_seconds
            retries = self.config.retries

            for i in range(repetitions):
                seed = seeds[i] if i < len(seeds) else 42
                payload = {"seed": seed}
                status = None
                output = None
                for attempt in range(retries + 1):
                    try:
                        if timeout is not None:
                            output = await asyncio.wait_for(
                                scenario_fn(runner, payload), timeout=timeout
                            )
                        else:
                            output = await scenario_fn(runner, payload)
                        status = "success"
                        break
                    except TimeoutError:
                        status = "timeout"
                        output = None
                        break  # Do not retry timeouts
                    except Exception:
                        if attempt < retries:
                            continue
                        status = "error"
                        output = None
                        break

                # Apply metrics
                metrics = {}
                if self.config.metrics:
                    metric_reg = MetricRegistry()
                    for metric_name in self.config.metrics:
                        try:
                            metric = metric_reg.create_metric(metric_name)
                            if metric:
                                metrics[metric_name] = (
                                    metric.calculate(output or {})
                                    if output is not None
                                    else 0.0
                                )
                        except Exception:
                            metrics[metric_name] = 0.0

                run_report = RunReport(
                    status=status,
                    output=output,
                    seed=seed,
                    metrics=metrics,
                )
                scenario_runs.append(run_report)

            # Compute aggregates for metrics
            aggregates = {}
            if scenario_runs and self.config.metrics:
                all_metrics = set()
                for r in scenario_runs:
                    all_metrics.update(r.metrics.keys())
                mean_metrics = {}
                for m in all_metrics:
                    values = [r.metrics.get(m, 0.0) for r in scenario_runs]
                    mean_metrics[m] = sum(values) / len(values)
                aggregates["metrics"] = {"mean": mean_metrics}

            # Apply validators
            if self.config.validators and scenario_runs:
                validations = []
                val_reg = ValidatorRegistry()
                run_data = [r.model_dump() for r in scenario_runs]
                for val_name in self.config.validators:
                    try:
                        validator = val_reg.create_validator(val_name)
                        if validator:
                            result = validator.validate(run_data)
                            result_dict = (
                                result.to_dict()
                                if hasattr(result, "to_dict")
                                else result
                            )
                            validations.append(
                                {"name": val_name, "result": result_dict}
                            )
                    except Exception:
                        validations.append(
                            {
                                "name": val_name,
                                "result": {
                                    "is_valid": False,
                                    "error": "Validation failed",
                                },
                            }
                        )
                aggregates["validations"] = validations

            scenario_report = ScenarioReport(
                key=scenario_spec.key,
                runs=scenario_runs,
                aggregates=aggregates,
            )
            scenario_reports.append(scenario_report)

        return EngineReport(scenario_reports=scenario_reports)
