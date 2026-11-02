"""User-requested chronological test cohorts, independent of batch size."""
import hashlib
import json

TEST_WINDOW_COUNTS = {"lamah_ce": 1425, "lamah_ice": 1440}


def select_test_windows(dataset, dataset_name, model_name):
    count = TEST_WINDOW_COUNTS[dataset_name]
    attribute = "ends" if hasattr(dataset, "ends") else "samples"
    indices = list(getattr(dataset, attribute))
    if len(indices) < count:
        raise ValueError(f"{dataset_name}: need {count} test windows, found {len(indices)}")
    selected = [int(value) for value in indices[:count]]
    if selected != sorted(selected) or len(set(selected)) != count:
        raise ValueError("Test windows must have unique chronological input-end indices")
    setattr(dataset, attribute, selected)
    return {"version": "fixed_chronological_test_windows_v1", "selection": "first chronological windows",
            "available_windows": len(indices), "selected_windows": count,
            "first_input_end_index": selected[0], "last_input_end_index": selected[-1],
            "input_end_indices_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
