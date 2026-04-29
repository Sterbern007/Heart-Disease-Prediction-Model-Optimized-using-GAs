from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from model import (
    build_pipeline,
    chromosome_key,
    crossover_hyperparameters,
    get_model_label,
    mutate_hyperparameters,
    sample_hyperparameters,
)


@dataclass
class EvaluatedCandidate:
    hyperparameters: dict[str, Any]
    validation_accuracy: float
    training_time: float
    time_score: float
    composite_fitness: float
    generation: int
    source: str
    pareto_rank: int = 999
    crowding_distance: float = 0.0

    def to_record(self, model_name: str) -> dict[str, Any]:
        record = {
            "generation": self.generation,
            "source": self.source,
            "validation_accuracy": self.validation_accuracy,
            "training_time": self.training_time,
            "time_score": self.time_score,
            "composite_fitness": self.composite_fitness,
            "pareto_rank": self.pareto_rank,
            "crowding_distance": self.crowding_distance,
            "model_label": get_model_label(model_name),
        }
        record.update(self.hyperparameters)
        return record


@dataclass
class OptimizationResult:
    model_name: str
    selected_candidate: EvaluatedCandidate
    pareto_front: list[EvaluatedCandidate]
    all_candidates: list[EvaluatedCandidate]
    history: list[dict[str, Any]]
    completed_generations: int
    stopped_early: bool


def calculate_time_score(training_time: float, baseline_training_time: float) -> float:
    baseline = max(baseline_training_time, 1e-8)
    return baseline / (baseline + training_time)


def calculate_composite_fitness(
    accuracy: float,
    training_time: float,
    baseline_training_time: float,
    accuracy_weight: float = 0.82,
) -> tuple[float, float]:
    time_score = calculate_time_score(training_time, baseline_training_time)
    fitness = (accuracy_weight * accuracy) + ((1 - accuracy_weight) * time_score)
    return time_score, fitness


def evaluate_hyperparameters(
    model_name: str,
    hyperparameters: dict[str, Any],
    x_train,
    x_validation,
    y_train,
    y_validation,
    baseline_training_time: float,
    generation: int,
    source: str,
    random_state: int = 42,
) -> EvaluatedCandidate:
    pipeline = build_pipeline(model_name, hyperparameters, random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start_time = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - start_time

    predictions = pipeline.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    time_score, fitness = calculate_composite_fitness(
        accuracy=accuracy,
        training_time=training_time,
        baseline_training_time=baseline_training_time,
    )

    return EvaluatedCandidate(
        hyperparameters=dict(hyperparameters),
        validation_accuracy=float(accuracy),
        training_time=float(training_time),
        time_score=float(time_score),
        composite_fitness=float(fitness),
        generation=generation,
        source=source,
    )


def evaluate_test_performance(
    model_name: str,
    hyperparameters: dict[str, Any],
    x_train,
    x_test,
    y_train,
    y_test,
    baseline_training_time: float,
    random_state: int = 42,
) -> dict[str, Any]:
    pipeline = build_pipeline(model_name, hyperparameters, random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start_time = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - start_time

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    time_score, fitness = calculate_composite_fitness(
        accuracy=accuracy,
        training_time=training_time,
        baseline_training_time=baseline_training_time,
    )

    return {
        "pipeline": pipeline,
        "test_accuracy": float(accuracy),
        "test_training_time": float(training_time),
        "test_time_score": float(time_score),
        "test_fitness": float(fitness),
    }


def dominates(candidate_a: EvaluatedCandidate, candidate_b: EvaluatedCandidate) -> bool:
    no_worse = (
        candidate_a.validation_accuracy >= candidate_b.validation_accuracy
        and candidate_a.training_time <= candidate_b.training_time
    )
    strictly_better = (
        candidate_a.validation_accuracy > candidate_b.validation_accuracy
        or candidate_a.training_time < candidate_b.training_time
    )
    return no_worse and strictly_better


def assign_pareto_metrics(candidates: list[EvaluatedCandidate]) -> list[list[EvaluatedCandidate]]:
    domination_counts = [0 for _ in candidates]
    dominated_indices: list[list[int]] = [[] for _ in candidates]
    fronts: list[list[int]] = [[]]

    for index_a, candidate_a in enumerate(candidates):
        candidate_a.crowding_distance = 0.0
        for index_b, candidate_b in enumerate(candidates):
            if index_a == index_b:
                continue
            if dominates(candidate_a, candidate_b):
                dominated_indices[index_a].append(index_b)
            elif dominates(candidate_b, candidate_a):
                domination_counts[index_a] += 1

        if domination_counts[index_a] == 0:
            candidate_a.pareto_rank = 0
            fronts[0].append(index_a)

    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front: list[int] = []
        for candidate_index in fronts[current_front]:
            for dominated_index in dominated_indices[candidate_index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    candidates[dominated_index].pareto_rank = current_front + 1
                    next_front.append(dominated_index)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    candidate_fronts = [[candidates[index] for index in front] for front in fronts if front]
    for front in candidate_fronts:
        assign_crowding_distance(front)
    return candidate_fronts


def assign_crowding_distance(front: list[EvaluatedCandidate]) -> None:
    if not front:
        return

    if len(front) <= 2:
        for candidate in front:
            candidate.crowding_distance = math.inf
        return

    for candidate in front:
        candidate.crowding_distance = 0.0

    for objective_name in ("validation_accuracy", "training_time"):
        ordered = sorted(front, key=lambda candidate: getattr(candidate, objective_name))
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf

        minimum = getattr(ordered[0], objective_name)
        maximum = getattr(ordered[-1], objective_name)
        if maximum == minimum:
            continue

        for index in range(1, len(ordered) - 1):
            if math.isinf(ordered[index].crowding_distance):
                continue
            previous_value = getattr(ordered[index - 1], objective_name)
            next_value = getattr(ordered[index + 1], objective_name)
            ordered[index].crowding_distance += (next_value - previous_value) / (maximum - minimum)


def sort_candidates(candidates: list[EvaluatedCandidate]) -> list[EvaluatedCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.pareto_rank,
            -candidate.crowding_distance,
            -candidate.composite_fitness,
            -candidate.validation_accuracy,
            candidate.training_time,
        ),
    )


class GeneticOptimizer:
    def __init__(
        self,
        model_name: str,
        x_train,
        x_validation,
        y_train,
        y_validation,
        population_size: int = 28,
        generations: int = 18,
        mutation_rate: float = 0.18,
        crossover_rate: float = 0.85,
        elitism_fraction: float = 0.15,
        stagnation_patience: int = 6,
        baseline_training_time: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.x_train = x_train
        self.x_validation = x_validation
        self.y_train = y_train
        self.y_validation = y_validation
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_fraction = elitism_fraction
        self.stagnation_patience = stagnation_patience
        self.baseline_training_time = baseline_training_time
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.evaluation_cache: dict[tuple[Any, ...], EvaluatedCandidate] = {}
        self.archive: dict[tuple[Any, ...], EvaluatedCandidate] = {}
        self.history: list[dict[str, Any]] = []

    def _evaluate(self, hyperparameters: dict[str, Any], generation: int, source: str) -> EvaluatedCandidate:
        key = chromosome_key(self.model_name, hyperparameters)
        if key in self.evaluation_cache:
            return replace(self.evaluation_cache[key], generation=generation, source=f"{source}-cached")

        candidate = evaluate_hyperparameters(
            model_name=self.model_name,
            hyperparameters=hyperparameters,
            x_train=self.x_train,
            x_validation=self.x_validation,
            y_train=self.y_train,
            y_validation=self.y_validation,
            baseline_training_time=self.baseline_training_time,
            generation=generation,
            source=source,
            random_state=self.random_state,
        )
        self.evaluation_cache[key] = candidate
        self.archive[key] = candidate
        return replace(candidate)

    def _initialize_population(self) -> list[EvaluatedCandidate]:
        population: list[EvaluatedCandidate] = []
        seen_keys: set[tuple[Any, ...]] = set()

        while len(population) < self.population_size:
            chromosome = sample_hyperparameters(self.model_name, self.rng)
            key = chromosome_key(self.model_name, chromosome)
            if key in seen_keys:
                continue
            population.append(self._evaluate(chromosome, generation=1, source="initial"))
            seen_keys.add(key)
        return population

    def _tournament_selection(self, population: list[EvaluatedCandidate], tournament_size: int = 3) -> EvaluatedCandidate:
        contenders = list(self.rng.choice(population, size=tournament_size, replace=False))
        ordered = sort_candidates(contenders)
        return ordered[0]

    def _breed_population(self, population: list[EvaluatedCandidate], generation: int) -> list[EvaluatedCandidate]:
        ordered_population = sort_candidates(population)
        elite_count = max(1, int(round(self.population_size * self.elitism_fraction)))
        next_generation = [
            replace(candidate, generation=generation, source="elite")
            for candidate in ordered_population[:elite_count]
        ]

        while len(next_generation) < self.population_size:
            parent_a = self._tournament_selection(ordered_population)
            parent_b = self._tournament_selection(ordered_population)

            if self.rng.random() < self.crossover_rate:
                child_hyperparameters = crossover_hyperparameters(
                    self.model_name,
                    parent_a.hyperparameters,
                    parent_b.hyperparameters,
                    self.rng,
                )
            else:
                child_hyperparameters = dict(parent_a.hyperparameters)

            child_hyperparameters = mutate_hyperparameters(
                self.model_name,
                child_hyperparameters,
                self.mutation_rate,
                self.rng,
            )

            next_generation.append(self._evaluate(child_hyperparameters, generation=generation, source="offspring"))

        return next_generation[: self.population_size]

    def _record_generation(self, generation: int, population: list[EvaluatedCandidate]) -> EvaluatedCandidate:
        fronts = assign_pareto_metrics(population)
        best_candidate = sort_candidates(population)[0]

        self.history.append(
            {
                "generation": generation,
                "best_accuracy": best_candidate.validation_accuracy,
                "best_training_time": best_candidate.training_time,
                "best_fitness": best_candidate.composite_fitness,
                "mean_accuracy": float(np.mean([candidate.validation_accuracy for candidate in population])),
                "mean_training_time": float(np.mean([candidate.training_time for candidate in population])),
                "pareto_size": len(fronts[0]),
            }
        )
        return best_candidate

    def run(self, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> OptimizationResult:
        population = self._initialize_population()
        best_candidate = self._record_generation(1, population)

        if progress_callback:
            progress_callback(
                {
                    "generation": 1,
                    "total_generations": self.generations,
                    "best_accuracy": best_candidate.validation_accuracy,
                    "best_training_time": best_candidate.training_time,
                    "best_fitness": best_candidate.composite_fitness,
                }
            )

        best_score = best_candidate.composite_fitness
        stagnant_generations = 0
        stopped_early = False

        for generation in range(2, self.generations + 1):
            population = self._breed_population(population, generation=generation)
            best_candidate = self._record_generation(generation, population)

            if best_candidate.composite_fitness > best_score + 1e-4:
                best_score = best_candidate.composite_fitness
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if progress_callback:
                progress_callback(
                    {
                        "generation": generation,
                        "total_generations": self.generations,
                        "best_accuracy": best_candidate.validation_accuracy,
                        "best_training_time": best_candidate.training_time,
                        "best_fitness": best_candidate.composite_fitness,
                    }
                )

            if stagnant_generations >= self.stagnation_patience:
                stopped_early = True
                break

        all_candidates = [replace(candidate) for candidate in self.archive.values()]
        pareto_fronts = assign_pareto_metrics(all_candidates)
        pareto_front = sort_candidates(pareto_fronts[0])
        selected_candidate = max(
            pareto_front,
            key=lambda candidate: (
                candidate.composite_fitness,
                candidate.validation_accuracy,
                -candidate.training_time,
            ),
        )

        return OptimizationResult(
            model_name=self.model_name,
            selected_candidate=selected_candidate,
            pareto_front=pareto_front,
            all_candidates=sort_candidates(all_candidates),
            history=self.history,
            completed_generations=len(self.history),
            stopped_early=stopped_early,
        )
