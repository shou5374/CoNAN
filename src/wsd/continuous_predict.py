import collections
import copy
import heapq
import os
import statistics
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional

import networkx as nx
import torch

from src.wsd.dataset import ConanSample
from src.wsd.dependency_finder import DependencyFinder
from src.scripts.predict import predict

from nltk.corpus import wordnet
from collections import defaultdict


def build_digraph_from_dependencies(wsd_instances_dependencies: Dict[str, List[str]]) -> nx.DiGraph:
    digraph = nx.DiGraph()
    for s, ts in wsd_instances_dependencies.items():
        digraph.add_node(s)
        for t in ts:
            digraph.add_edge(s, t)
    return digraph


def contains_cycles(wsd_instances_dependencies: Dict[str, List[str]]) -> bool:
    digraph = build_digraph_from_dependencies(wsd_instances_dependencies)
    try:
        cycle = nx.find_cycle(digraph)
        print(cycle)
        return True
    except nx.NetworkXNoCycle:
        return False


def report_predictions(
    output_file,
    predicted_conan_samples: List[Tuple[ConanSample, int]],
    depends_on_samples: Dict[str, Tuple[ConanSample, int]],
    wordnet_sense_inventory,
):
    with open(output_file, "w") as f:
        for sample, prediction in predicted_conan_samples:
            if "unannotated" in sample.sample_id:
                continue
            f.write(f"# instance id: {sample.sample_id}\n")
            # f.write(f"# marked text: {sample.marked_text}\n")
            f.write("# origin text: {}\n".format(" ".join(sample.marked_text)))
            f.write(f"# context definitions:\n")
            for d, p in sample.context_definitions:
                f.write(f"   - {d.linker}@{p} \t {d.text}\n")
            f.write(f"# candidate definitions:\n")
            for i, d in enumerate(sample.candidate_definitions):
                count = wordnet_sense_inventory.get_gloss_count(d.text)
                sense = wordnet_sense_inventory.get_sense_from_gloss(d.text)
                synset = wordnet.lemma_from_key(sense).synset()
                hypernym_synset = synset.hypernyms()[:1]
                if len(hypernym_synset) > 0:
                    hypernyms_sense = hypernym_synset[0].lemmas()[0].key()
                else:
                    hypernyms_sense = "None"
                gold_marker = "G" if d.text in set([gd.text for gd in sample.gold_definitions]) else " "
                if i == prediction:
                    f.write(f" {gold_marker} * {d.linker} \t {count} \t {hypernyms_sense} \t {d.text}\n")
                else:
                    f.write(f"{gold_marker} - {d.linker} \t {count} \t {hypernyms_sense} \t {d.text}\n")
            f.write("\n")

    if depends_on_samples is not None:
        other_infos = defaultdict(int)
        with open(output_file.replace(".report", ".error.report"), "w") as f:
            for sample, prediction in predicted_conan_samples:
                if "unannotated" in sample.sample_id:
                    continue
                pos_count = 0
                for tmp_pos in ["n", "v", "a", "r"]:
                    if len(wordnet_sense_inventory.get_possible_senses(sample.disambiguation_instance.lemma, tmp_pos)) > 0:
                        pos_count += 1
                for i, d in enumerate(sample.candidate_definitions):
                    if d.text in set([gd.text for gd in sample.gold_definitions]) and i == prediction:
                        if len(sample.candidate_definitions) > 1:
                            other_infos["t_pos_count_over_1" if pos_count > 1 else "t_pos_count_is_1"] += 1
                        break
                else:
                    if len(sample.candidate_definitions) > 1:
                        other_infos["f_pos_count_over_1" if pos_count > 1 else "f_pos_count_is_1"] += 1
                    f.write(f"# instance id: {sample.sample_id}\n")
                    # f.write(f"# marked text: {sample.marked_text}\n")
                    f.write("# origin text: {}\n".format(" ".join(sample.marked_text)))
                    f.write(f"# context definitions:\n")
                    for i, d in enumerate(sample.candidate_definitions):
                        count = wordnet_sense_inventory.get_gloss_count(d.text)
                        sense = wordnet_sense_inventory.get_sense_from_gloss(d.text)
                        synset = wordnet.lemma_from_key(sense).synset()
                        pos = synset.pos()
                        if pos == "s":
                            pos = "a"
                        hypernym_synset = synset.hypernyms()[:1]
                        if len(hypernym_synset) > 0:
                            hypernyms_sense = hypernym_synset[0].lemmas()[0].key()
                        else:
                            hypernyms_sense = "None"
                        gold_marker = "G" if d.text in set([gd.text for gd in sample.gold_definitions]) else " "
                        if i == prediction:
                            other_infos[pos] += 1
                            other_infos[f"{pos}_ncd"] += len(sample.candidate_definitions)
                            f.write(f" {gold_marker} * {d.linker} \t {count} \t {hypernyms_sense} \t {d.text}\n")
                        else:
                            f.write(f"{gold_marker} - {d.linker} \t {count} \t {hypernyms_sense} \t {d.text}\n")
                    f.write(f"# depends on definitions:\n")
                    for dep_sample, dep_pred in depends_on_samples[sample.sample_id]:
                        dep_pred_def = dep_sample.candidate_definitions[dep_pred]
                        gold_marker = "G" if dep_pred_def in dep_sample.gold_definitions else " "
                        f.write(f" ({gold_marker}) {dep_pred_def.linker} \t {dep_pred_def.text}\n")
                f.write("\n")
            for pos in ["n", "v", "a", "r"]:
                other_infos[f"{pos}_ncd"] = other_infos[f"{pos}_ncd"] / (other_infos[pos] + 1e-8)
            print(other_infos, file=f)


class Predictor:
    def predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        wordnet_sense_inventory=None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:
        predicted_conan_samples, depends_on_samples = self._predict(
            conan_samples,
            already_kwown_predictions=already_kwown_predictions,
            reporting_folder=reporting_folder,
            **kwargs,
        )
        if reporting_folder is not None:
            report_predictions(
                f"{reporting_folder}/predictions.report",
                predicted_conan_samples,
                depends_on_samples,
                wordnet_sense_inventory,
            )
        return predicted_conan_samples

    def _predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:
        raise NotImplementedError


class TeacherForcedPredictor(Predictor):
    def __init__(self, dependency_finder: DependencyFinder):
        self.dependency_finder = dependency_finder

    def _predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:

        assert already_kwown_predictions is None, "already_kwown_predictions is not None on TeacherForcedPredictor (doesn't make any sense"

        # reset deps if they were set and compute instance_id2sample mapping
        instance_id2sample = {}
        for sample in conan_samples:
            if sample.sample_id is not None:
                assert sample.sample_id not in instance_id2sample
                instance_id2sample[sample.sample_id] = sample
                sample.reset_context_definitions()

        # apply dependency finder

        # compute and assign dependencies
        dep_adj_l = {}
        for sample in conan_samples:
            instance_id = sample.sample_id
            if instance_id is None:
                continue
            instance_id2sample[instance_id] = sample
            sample_deps = self.dependency_finder.find_dependencies(
                sample.kwargs["enlarged_disambiguation_context"],
                sample.kwargs["enlarged_disambiguation_index"],
            )
            dep_adj_l[instance_id] = [sd.instance_id for sd in sample_deps]

        # check no cycles have been created
        assert not contains_cycles(dep_adj_l)

        # set context definitions
        for sample in conan_samples:
            instance_id = sample.sample_id
            for _iid in dep_adj_l[instance_id]:
                _s = instance_id2sample[_iid]
                sample.context_definitions.append((_s.gold_definitions[0], sample.get_sample_id_position(_iid)))

        # predict

        predictions: Dict[str, int] = {}

        for sample, probs in predict(samples=conan_samples, **kwargs):
            predictions[sample.sample_id] = torch.tensor(probs).argmax().item()

        # return
        return [(sample, predictions[sample.sample_id]) for sample in conan_samples], None


class GreedyDepPredictor(Predictor):
    def __init__(self, dependency_finder: DependencyFinder):
        self.dependency_finder = dependency_finder

    def _predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:

        # reset deps if they were set and compute instance_id2sample mapping
        instance_id2sample = {}
        for sample in conan_samples:
            if sample.sample_id is not None:
                assert sample.sample_id not in instance_id2sample
                instance_id2sample[sample.sample_id] = sample
                sample.reset_context_definitions()

        # apply dependency finder

        depends_on = {}

        # compute and assign dependencies

        dep_adj_l = {}

        for sample in conan_samples:

            instance_id = sample.sample_id
            if instance_id is None:
                continue

            sample_deps = self.dependency_finder.find_dependencies(
                sample.kwargs["enlarged_disambiguation_context"],
                sample.kwargs["enlarged_disambiguation_index"],
            )
            dep_adj_l[instance_id] = [sd.instance_id for sd in sample_deps]

        # check no cycles have been created
        assert not contains_cycles(dep_adj_l)
        depends_on.update(**dep_adj_l)

        # do rounds

        done = set()
        predictions: Dict[str, int] = {}

        if already_kwown_predictions is not None:
            for k, v in already_kwown_predictions.items():
                predictions[k] = v
                done.add(k)

        while len(done) != len(depends_on):

            # compute round samples

            round_samples = []

            for instance_id, sample in instance_id2sample.items():

                # check if sample can be done
                if instance_id in done or any(_iid not in done for _iid in depends_on[instance_id]):
                    continue

                # populate context definitions
                assert len(sample.context_definitions) == 0
                for _iid in depends_on[instance_id]:
                    _s = instance_id2sample[_iid]
                    _p = predictions[_iid]
                    sample.context_definitions.append(
                        (
                            _s.candidate_definitions[_p],
                            sample.in_context_sample_id2position[_iid],
                        )
                    )

                # add to round samples
                round_samples.append(sample)

            # predict

            # print(f"Round samples: {len(round_samples)}")
            for sample, probs in predict(samples=round_samples, **kwargs):
                predictions[sample.sample_id] = torch.tensor(probs).argmax().item()

            # update done
            done.update([sample.sample_id for sample in round_samples])

        depends_on_samples = {
            instance_id: [(instance_id2sample[dep_sample_id], predictions[dep_sample_id]) for dep_sample_id in dep_sample_ids]
            for instance_id, dep_sample_ids in depends_on.items()
        }

        # return
        return [(sample, predictions[sample.sample_id]) for sample in conan_samples], depends_on_samples


@dataclass
class _Beam:
    sub_beams: List[Tuple[List[int], float]]
    beam_path: List[str]
    position: int

    def is_finished(self) -> bool:
        return self.position >= len(self.beam_path)

    def get_n_remaining(self) -> int:
        return len(self.beam_path) - self.position

    def get_next(self) -> str:
        return self.beam_path[self.position]


class BeamDepPredictor(Predictor):
    def __init__(
        self,
        dependency_finder: DependencyFinder,
        beam_size: int,
        enable_reporting: bool = False,
    ):
        self.dependency_finder = dependency_finder
        self.beam_size = beam_size
        self.enable_reporting = enable_reporting

    def _predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:

        if already_kwown_predictions is not None:
            raise NotImplementedError

        # reset deps if they were set and compute instance_id2sample mapping
        instance_id2sample = {}
        for sample in conan_samples:
            if sample.sample_id is not None:
                assert sample.sample_id not in instance_id2sample
                instance_id2sample[sample.sample_id] = sample
                sample.reset_context_definitions()

        # apply dependency finder
        dep_adj_l = {}
        for sample in conan_samples:
            instance_id = sample.sample_id
            if instance_id is not None:
                sample_deps = self.dependency_finder.find_dependencies(
                    sample.kwargs["enlarged_disambiguation_context"],
                    sample.kwargs["enlarged_disambiguation_index"],
                )
                dep_adj_l[instance_id] = [sd.instance_id for sd in sample_deps]

        # check no cycles have been created
        assert not contains_cycles(dep_adj_l)
        depends_on = dep_adj_l

        # divide samples in connected componenets (and beams)
        instance_id2beam_id, beams = {}, []
        digraph = build_digraph_from_dependencies(dep_adj_l)
        for cc in nx.weakly_connected_components(digraph):
            beam_id = len(beams)
            # compute instance -> beam mapping
            for instance_id in cc:
                instance_id2beam_id[instance_id] = beam_id
            # compute beam path
            beam_path = self.compute_beam_path(cc, depends_on)
            # add beam
            beams.append(_Beam(sub_beams=[([], 0.0)], beam_path=beam_path, position=0))

        # if reporting is enabled, create a reporting file for each beam
        beam_id2reporting_file = None
        if self.enable_reporting:
            os.mkdir(f"{reporting_folder}/beams")
            beam_id2reporting_file = {i: open(f"{reporting_folder}/beams/{i}", "w") for i, _ in enumerate(beams)}

        # do beam

        visited = set()

        while len(visited) != len(depends_on):

            # build round samples
            round_samples = []
            beams_active = 0
            for i, beam in enumerate(beams):
                if beam.is_finished():
                    continue
                instance_id = beam.get_next()
                beams_active += 1
                for j, (sub_beam, _) in enumerate(beam.sub_beams):
                    sample = copy.deepcopy(instance_id2sample[instance_id])
                    sample.kwargs["beam-search"] = i, j
                    round_samples.append(sample)
                    _iid2_p_idx = {_iid: _p_idx for _iid, _p_idx in zip(beam.beam_path, sub_beam)}
                    for _iid in depends_on[instance_id]:
                        _s = instance_id2sample[_iid]
                        _p_idx = _iid2_p_idx[_iid]
                        sample.context_definitions.append(
                            (
                                _s.candidate_definitions[_p_idx],
                                sample.get_sample_id_position(_iid),
                            )
                        )

            # predict and group beams
            # print(f"# round samples: {len(round_samples)}")
            print(f"# beams active: {beams_active}")
            print(f"# avg beam length remaining: {statistics.mean([beam.get_n_remaining() for beam in beams if not beam.is_finished()])}")
            beam_id2predictions = collections.defaultdict(list)
            for sample, probs in predict(samples=round_samples, **kwargs):
                i, j = sample.kwargs["beam-search"]
                beam_id2predictions[i].append((j, sample, probs))

            # process and update beams
            for beam_id, beam_predictions in beam_id2predictions.items():

                # retrieve beam
                beam = beams[beam_id]

                # rebuild sub beams
                sub_beams = []
                for j, sample, probs in beam_predictions:
                    history, history_score = beam.sub_beams[j]
                    log_probs = torch.tensor(probs).log()
                    predicted_idxs = log_probs.argsort(descending=True)
                    for idx in predicted_idxs:
                        sub_beams.append(
                            (
                                history + [idx.item()],
                                history_score + log_probs[idx].item(),
                            )
                        )

                # extract best beams
                best_sub_beams_idx = heapq.nlargest(self.beam_size, range(len(sub_beams)), key=lambda x: sub_beams[x][1])
                # best_sub_beams = heapq.nlargest(self.beam_size, sub_beams, key=lambda x: x[1])

                # report
                if beam_id2reporting_file is not None:
                    rf = beam_id2reporting_file[beam_id]
                    rf.write(f"# beam path:\n")
                    for n in beam.beam_path:
                        rf.write(f' {">" if n == sample.sample_id else " "}{n}\n')
                    rf.write(f"# beams\n")
                    for i, (history, history_score) in enumerate(sub_beams):
                        rf.write(f' {">" if i in best_sub_beams_idx else " "}{history_score:.4f}\n')
                        for _iid, _p_idx in zip(beam.beam_path, history):
                            _s = instance_id2sample[_iid]
                            gold_marker = "!" if _s.candidate_definitions[_p_idx] in _s.gold_definitions else " "
                            rf.write(f"  {gold_marker} * {_s.candidate_definitions[_p_idx].linker} \t {_s.candidate_definitions[_p_idx].text}\n")
                    rf.write("\n")

                # update beam position
                beam.sub_beams = [sub_beams[idx] for idx in best_sub_beams_idx]
                beam.position += 1

            # update done
            visited.update([sample.sample_id for sample in round_samples])

        # build predictions map
        predictions = {}
        for beam in beams:
            best_sub_beam = max(beam.sub_beams, key=lambda x: x[1])[0]
            assert len(beam.beam_path) == len(best_sub_beam)
            for _id, _p_idx in zip(beam.beam_path, best_sub_beam):
                predictions[_id] = _p_idx

        # close reporting files
        if beam_id2reporting_file is not None:
            for _, v in beam_id2reporting_file.items():
                v.close()

        # return
        return [(sample, predictions[sample.sample_id]) for sample in conan_samples], None

    def compute_beam_path(self, connected_component: Set[str], depends_on: Dict[str, List[str]]) -> List[str]:
        beam_path, added = [], set()
        while len(beam_path) != len(connected_component):
            for instance_id in connected_component:
                if instance_id in added or any(_iid not in added for _iid in depends_on[instance_id]):
                    continue
                beam_path.append(instance_id)
                added.add(instance_id)
        return beam_path


class BalancingPredictor(Predictor):
    def __init__(self, dependency_finder: DependencyFinder, predictor: Predictor):
        self.dependency_finder = dependency_finder
        self.predictor = predictor

    def _predict(
        self,
        conan_samples: List[ConanSample],
        already_kwown_predictions: Optional[Dict[str, int]] = None,
        reporting_folder: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[ConanSample, int]]:

        # base predictor predictions indexing
        predicted_conan_samples = self.predictor.predict(conan_samples, **kwargs)
        predictions = {cs.sample_id: pi for cs, pi in predicted_conan_samples}

        # reset deps if they were set and compute instance_id2sample mapping
        instance_id2sample = {}
        for sample in conan_samples:
            if sample.sample_id is not None:
                assert sample.sample_id not in instance_id2sample
                instance_id2sample[sample.sample_id] = sample
                sample.reset_context_definitions()

        # apply dependency finder

        depends_on = {}

        # compute and assign dependencies

        dep_adj_l = {}

        for sample in conan_samples:

            instance_id = sample.sample_id
            if instance_id is None:
                continue

            sample_deps = self.dependency_finder.find_dependencies(
                sample.kwargs["enlarged_disambiguation_context"],
                sample.kwargs["enlarged_disambiguation_index"],
            )
            dep_adj_l[instance_id] = [sd.instance_id for sd in sample_deps]

        depends_on.update(**dep_adj_l)

        round_count = 1
        while True:

            round_predictions = {}

            for instance_id, sample in instance_id2sample.items():

                # populate context definitions with predictions from the last round
                sample.reset_context_definitions()
                for _iid in depends_on[instance_id]:
                    _s = instance_id2sample[_iid]
                    _p = predictions[_iid]
                    sample.context_definitions.append(
                        (
                            _s.candidate_definitions[_p],
                            sample.get_sample_id_position(_iid),
                        )
                    )

            print(f"Balancing round {round_count} starting")
            for sample, probs in predict(samples=list(instance_id2sample.values()), **kwargs):
                round_predictions[sample.sample_id] = torch.tensor(probs).argmax().item()

            changed_instances = [
                (iid, round_predictions[iid], predictions[iid]) for iid in round_predictions if round_predictions[iid] != predictions[iid]
            ]

            if len(changed_instances) > 0:
                predictions = round_predictions
                print(f"Number of changed instances: {len(changed_instances)}")
            else:
                break

            round_count += 1

            if round_count == 3:  # todo: remove
                break

        return [(sample, predictions[sample.sample_id]) for sample in conan_samples], None
