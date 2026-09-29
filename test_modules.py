"""
MnBrain - Module Test Suite
=============================
Run with:  python test_modules.py

Covers:
  Module 1              - Geospatial AI (Ordinary Kriging block model)
  Module 2              - Optimization Engine (LP grade blending)
  Data ingestion layer  - CSV/Excel parsing, geological & stockpile validation
  Uploaded-data kriging - Deterministic block model from a sample table
  Constraint validation - verify_blend re-checks the solver output
  Upload workflow       - sample_geological_data.csv end-to-end regression
  Grid cap regression   - wide extents never exceed MAX_GRID_CELLS per axis
  Sample model quality  - the bundled sample yields a meaningful 3D model
  KPI state consistency - uploaded data drives every displayed KPI
Exit code is non-zero if any test fails.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from data_loader import (DataLoadError,
                         load_demonstration_geological_samples,
                         load_demonstration_stockpiles, parse_upload,
                         validate_geological_data, validate_stockpile_data)
from geospatial_model import (MAX_GRID_CELLS, get_demonstration_samples,
                              get_kriged_block_model, krige_uploaded_samples,
                              list_pits)
from optimization_engine import (REQUIRED_COLUMNS, calculate_optimal_blend,
                                 load_sample_stockpiles, verify_blend)
from app import DEFAULT_CUTOFF, DEFAULT_DEPTH, DEFAULT_PIT

PIT = list_pits()[0]
OTHER_PIT = list_pits()[1]
SAMPLE_GEO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sample_geological_data.csv")


def _open_sample_csv():
    """Expose the bundled sample CSV as bytes, like a Streamlit upload."""
    with open(SAMPLE_GEO_CSV, "rb") as fh:
        return io.BytesIO(fh.read())


# ===========================================================================
# Module 1 - Geospatial AI
# ===========================================================================

class TestGeospatialModel(unittest.TestCase):

    def setUp(self):
        self.pit = PIT
        self.depth = 50
        self.cutoff = 3.0

    def test_valid_pit_returns_dataframe(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_required_columns(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        for col in ["X", "Y", "Z", "Mn_Grade", "Fe_Grade", "UNFC_Code"]:
            self.assertIn(col, df.columns, f"missing column {col}")

    def test_additional_columns(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        for col in ["Block_ID", "Kriging_Variance", "Confidence"]:
            self.assertIn(col, df.columns, f"missing column {col}")

    def test_no_nan_values(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        self.assertFalse(df.isna().any().any(), "NaN values found in model")

    def test_reproducibility(self):
        a = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        b = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        pd.testing.assert_frame_equal(a, b)

    def test_different_pits_differ(self):
        a = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        b = get_kriged_block_model(OTHER_PIT, self.depth, self.cutoff)
        self.assertFalse(np.allclose(a["Mn_Grade"].values,
                                     b["Mn_Grade"].values),
                         "different pits produced identical grades")

    def test_all_pits_valid(self):
        for pit in list_pits():
            df = get_kriged_block_model(pit, 50, 3.0)
            self.assertGreater(len(df), 0, f"empty model for {pit}")

    def test_depth_controls_block_count(self):
        shallow = get_kriged_block_model(self.pit, 10, self.cutoff)
        deep = get_kriged_block_model(self.pit, 100, self.cutoff)
        self.assertGreater(len(deep), len(shallow),
                           "deeper model must have more blocks")

    def test_grade_range_mn(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        self.assertTrue((df["Mn_Grade"] >= 20).all())
        self.assertTrue((df["Mn_Grade"] <= 55).all())

    def test_grade_range_fe(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        self.assertTrue((df["Fe_Grade"] >= 5).all())
        self.assertTrue((df["Fe_Grade"] <= 35).all())

    def test_unfc_codes_valid(self):
        df = get_kriged_block_model(self.pit, self.depth, self.cutoff)
        allowed = {"111", "UNVERIFIED"}
        self.assertTrue(set(df["UNFC_Code"].unique()) <= allowed)

    def test_confidence_threshold_effect(self):
        low = get_kriged_block_model(self.pit, self.depth, 1.5)
        high = get_kriged_block_model(self.pit, self.depth, 45.0)
        self.assertGreaterEqual((high["UNFC_Code"] == "111").sum(),
                                (low["UNFC_Code"] == "111").sum(),
                                "higher threshold must verify at least as many")

    def test_invalid_depth_low(self):
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, 5, self.cutoff)

    def test_invalid_depth_high(self):
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, 150, self.cutoff)

    def test_invalid_depth_type(self):
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, "deep", self.cutoff)

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, self.depth, 0.0)
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, self.depth, -3.0)
        with self.assertRaises(ValueError):
            get_kriged_block_model(self.pit, self.depth, 150.0)

    def test_invalid_pit(self):
        with self.assertRaises(ValueError):
            get_kriged_block_model("Nonexistent Pit", self.depth, self.cutoff)


# ===========================================================================
# Module 2 - Optimization Engine
# ===========================================================================

class TestOptimizationEngine(unittest.TestCase):

    def setUp(self):
        self.stockpiles = load_sample_stockpiles()
        self.tonnage = 10000.0
        self.mn = 42.0
        self.fe = 20.0

    def test_valid_input_columns(self):
        self.assertEqual(set(REQUIRED_COLUMNS),
                         set(self.stockpiles.columns))

    def test_missing_column(self):
        df = self.stockpiles.drop(columns=["Fe_Grade"])
        result = calculate_optimal_blend(df, self.tonnage, self.mn, self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")
        self.assertIn("Fe_Grade", result["message"])

    def test_empty_input(self):
        result = calculate_optimal_blend(pd.DataFrame(), self.tonnage,
                                         self.mn, self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")

    def test_successful_optimization(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertTrue(result["success"], result.get("message"))
        self.assertEqual(result["status"], "Optimal")

    def test_tonnage_constraint(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertAlmostEqual(result["total_tonnage"], self.tonnage, places=3)

    def test_mn_constraint(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertGreaterEqual(result["blended_mn"] + 1e-6, self.mn)

    def test_fe_constraint(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertLessEqual(result["blended_fe"] - 1e-6, self.fe)

    def test_availability_constraint(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        alloc = result["allocations"]
        self.assertTrue((alloc >= -1e-6).all())
        for j in range(len(self.stockpiles)):
            self.assertLessEqual(alloc[j] - 1e-6,
                                 float(self.stockpiles["Available_Tons"].iloc[j]))

    def test_allocation_sums_to_target(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertAlmostEqual(result["allocations"].sum(), self.tonnage,
                               places=2)

    def test_infeasible_tonnage(self):
        result = calculate_optimal_blend(self.stockpiles, 50000.0, self.mn,
                                         self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Infeasible")
        self.assertIn("tonnage", result["message"].lower())

    def test_infeasible_mn(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage, 62.0,
                                         self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Infeasible")
        self.assertIn("Mn", result["message"])

    def test_infeasible_fe(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, 3.0)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Infeasible")
        self.assertIn("Fe", result["message"])

    def test_invalid_target_negative_tonnage(self):
        result = calculate_optimal_blend(self.stockpiles, -1000.0, self.mn,
                                         self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")

    def test_invalid_target_mn(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage, 150.0,
                                         self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")

    def test_invalid_target_types(self):
        result = calculate_optimal_blend(self.stockpiles, "many", self.mn,
                                         self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")

    def test_negative_availability(self):
        df = self.stockpiles.copy()
        df.loc[0, "Available_Tons"] = -100
        result = calculate_optimal_blend(df, self.tonnage, self.mn, self.fe)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Invalid Input")

    def test_result_table_present(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        self.assertIsNotNone(result["result_df"])
        cols = result["result_df"].columns
        for col in ["Stockpile_ID", "Allocated_Tons", "Mn_Grade", "Fe_Grade",
                    "Cost_per_Ton", "Utilization_Pct"]:
            self.assertIn(col, cols)

    def test_no_exceptions_propagation(self):
        # The engine must return a result dict, never raise.
        for _ in range(3):
            case = calculate_optimal_blend(pd.DataFrame({"x": [1]}), 0, 0, 0)
            self.assertIsInstance(case, dict)
        self.assertEqual(case["status"], "Invalid Input")


# ===========================================================================
# Module 3 - Data Ingestion & Validation
# ===========================================================================

class TestDataLoader(unittest.TestCase):

    def test_demo_geological_valid(self):
        df = load_demonstration_geological_samples(PIT)
        result = validate_geological_data(df)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["summary"]["n_valid"], len(df))
        self.assertEqual(result["cleaned"].shape, df.shape)

    def test_geo_column_aliases_normalised(self):
        df = load_demonstration_geological_samples(PIT)
        renamed = df.rename(columns={"X": "Easting", "Y": "Northing",
                                     "Depth": "depth_m",
                                     "Mn_Grade": "Mn_pct",
                                     "Fe_Grade": "Fe%"})
        result = validate_geological_data(renamed)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("X", result["cleaned"].columns)

    def test_geo_missing_column_error(self):
        df = load_demonstration_geological_samples(PIT).drop(columns=["Fe_Grade"])
        result = validate_geological_data(df)
        self.assertFalse(result["ok"])
        self.assertIn("Fe_Grade", result["errors"][0])

    def test_geo_negative_depth_error(self):
        df = load_demonstration_geological_samples(PIT).copy()
        df.loc[0, "Depth"] = -5
        result = validate_geological_data(df)
        self.assertFalse(result["ok"])
        self.assertIn("negative depths", result["errors"][0])

    def test_geo_out_of_range_grade_error(self):
        df = load_demonstration_geological_samples(PIT).copy()
        df.loc[0, "Mn_Grade"] = 120.0
        result = validate_geological_data(df)
        self.assertFalse(result["ok"])
        self.assertIn("Mn", result["errors"][0])

    def test_geo_duplicates_removed_with_warning(self):
        df = pd.concat([
            load_demonstration_geological_samples(PIT),
            load_demonstration_geological_samples(PIT).iloc[:3],
        ], ignore_index=True)
        result = validate_geological_data(df)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["summary"]["n_duplicates"], 3)
        self.assertEqual(result["summary"]["n_valid"],
                         len(load_demonstration_geological_samples(PIT)))
        self.assertTrue(any("duplicate" in w.lower() for w in result["warnings"]))

    def test_geo_inf_values_treated_as_missing(self):
        df = load_demonstration_geological_samples(PIT).copy()
        df.loc[0, "X"] = np.inf
        result = validate_geological_data(df)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["summary"]["n_missing_values"], 1)

    def test_geo_empty_frame_error(self):
        result = validate_geological_data(pd.DataFrame())
        self.assertFalse(result["ok"])

    def test_geo_non_table_error(self):
        self.assertFalse(validate_geological_data(None)["ok"])

    def test_stockpile_demo_valid(self):
        df = load_demonstration_stockpiles()
        result = validate_stockpile_data(df)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["summary"]["n_valid"], len(df))

    def test_stockpile_negative_cost_error(self):
        df = load_demonstration_stockpiles().copy()
        df.loc[0, "Cost_per_Ton"] = -50.0
        result = validate_stockpile_data(df)
        self.assertFalse(result["ok"])
        self.assertIn("cost", result["errors"][0].lower())

    def test_stockpile_negative_availability_error(self):
        df = load_demonstration_stockpiles().copy()
        df.loc[0, "Available_Tons"] = -10.0
        result = validate_stockpile_data(df)
        self.assertFalse(result["ok"])

    def test_stockpile_duplicate_ids_removed(self):
        df = pd.concat([
            load_demonstration_stockpiles(),
            load_demonstration_stockpiles().iloc[[0]],
        ], ignore_index=True)
        result = validate_stockpile_data(df)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["summary"]["n_duplicates"], 1)
        self.assertTrue(any("duplicate" in w.lower() for w in result["warnings"]))

    def test_stockpile_missing_column_error(self):
        df = load_demonstration_stockpiles().drop(columns=["Fe_Grade"])
        result = validate_stockpile_data(df)
        self.assertFalse(result["ok"])
        self.assertIn("Fe_Grade", result["errors"][0])

    def test_parse_upload_csv(self):
        df = load_demonstration_geological_samples(PIT)
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        parsed = parse_upload(buf, "samples.csv")
        pd.testing.assert_frame_equal(parsed, df)

    def test_parse_upload_excel(self):
        df = load_demonstration_stockpiles()
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        parsed = parse_upload(buf, "stock.xlsx")
        pd.testing.assert_frame_equal(parsed, df)

    def test_parse_upload_unsupported_extension(self):
        with self.assertRaises(DataLoadError):
            parse_upload(io.BytesIO(b"x"), "notes.pdf")

    def test_parse_upload_semicolon_csv(self):
        df = load_demonstration_geological_samples(PIT)
        text = df.to_csv(index=False, sep=";")
        parsed = parse_upload(io.BytesIO(text.encode("utf-8")), "data.csv")
        self.assertEqual(len(parsed), len(df))
        self.assertIn("Mn_Grade", parsed.columns)


# ===========================================================================
# Module 4 - Kriging of uploaded sample tables
# ===========================================================================

class TestUploadedKriging(unittest.TestCase):

    def test_standard_schema_output(self):
        samples = get_demonstration_samples(PIT)
        model = krige_uploaded_samples(samples, 50, 3.0)
        for col in ["Block_ID", "X", "Y", "Z", "Mn_Grade", "Fe_Grade",
                    "Kriging_Variance", "Confidence", "UNFC_Code"]:
            self.assertIn(col, model.columns, f"missing {col}")
        self.assertGreater(len(model), 0)
        self.assertFalse(model.isna().any().any())

    def test_reproducible(self):
        samples = get_demonstration_samples(PIT)
        a = krige_uploaded_samples(samples, 50, 3.0)
        b = krige_uploaded_samples(samples, 50, 3.0)
        pd.testing.assert_frame_equal(a, b)

    def test_matches_demo_path_when_same_extent(self):
        # Feeding the demo samples through the uploaded path must reproduce
        # the demonstration grade field closely (same samples, same
        # Ordinary-Kriging algorithm). The uploaded grid is derived from the
        # observed sample extent rather than the fixed demo box, so exact
        # equality is not required - only statistical agreement.
        demo = get_kriged_block_model(PIT, 50, 3.0)
        up = krige_uploaded_samples(get_demonstration_samples(PIT), 50, 3.0,
                                    nugget=0.3, partial_sill=32.0,
                                    range_m=1000.0, anisotropy_z=2.0)
        self.assertGreater(len(up), 0)
        self.assertAlmostEqual(demo["Mn_Grade"].mean(), up["Mn_Grade"].mean(),
                               delta=0.5)
        self.assertAlmostEqual(demo["Fe_Grade"].mean(), up["Fe_Grade"].mean(),
                               delta=0.5)
        allowed = {"111", "UNVERIFIED"}
        self.assertEqual(set(up["UNFC_Code"].unique()) & allowed,
                         set(up["UNFC_Code"].unique()))

    def test_unfc_screening_terminology(self):
        up = krige_uploaded_samples(get_demonstration_samples(PIT), 50, 3.0)
        allowed = {"111", "UNVERIFIED"}
        self.assertEqual(set(up["UNFC_Code"].unique()) & allowed,
                         set(up["UNFC_Code"].unique()))

    def test_invalid_low_depth_rejected(self):
        samples = get_demonstration_samples(PIT)
        with self.assertRaises(ValueError):
            krige_uploaded_samples(samples, 5, 3.0)

    def test_empty_samples_rejected(self):
        with self.assertRaises(ValueError):
            krige_uploaded_samples(pd.DataFrame(), 50, 3.0)

    def test_missing_column_rejected(self):
        samples = get_demonstration_samples(PIT).drop(columns=["Fe_Grade"])
        with self.assertRaises(ValueError):
            krige_uploaded_samples(samples, 50, 3.0)

    def test_degenerate_extent_padded(self):
        small = get_demonstration_samples(PIT).iloc[:30].copy()
        small.loc[:, "X"] = 100.0
        small.loc[:, "Y"] = 100.0
        model = krige_uploaded_samples(small, 20, 3.0)
        self.assertGreater(len(model), 0)


# ===========================================================================
# Module 5 - Constraint Verification (verify_blend)
# ===========================================================================

class TestConstraintValidation(unittest.TestCase):

    def setUp(self):
        self.stockpiles = load_sample_stockpiles()
        self.tonnage = 10000.0
        self.mn = 42.0
        self.fe = 20.0

    def test_feasible_blend_passes_all_checks(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        v = verify_blend(result, self.tonnage, self.mn, self.fe)
        self.assertTrue(v["all_pass"], v)
        self.assertGreaterEqual(len(v["checks"]), 5)
        for c in v["checks"]:
            self.assertEqual(c["status"], "PASS")
        self.assertTrue(all(k in c for k in
                            ("constraint", "required", "achieved", "status"))
                        for c in v["checks"])

    def test_mn_violation_detected(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        v = verify_blend(result, self.tonnage, self.mn + 5.0, self.fe)
        self.assertFalse(v["all_pass"])
        mn_check = next(c for c in v["checks"]
                        if c["constraint"].startswith("Minimum"))
        self.assertEqual(mn_check["status"], "FAIL")

    def test_tonnage_violation_detected(self):
        result = calculate_optimal_blend(self.stockpiles, self.tonnage,
                                         self.mn, self.fe)
        v = verify_blend(result, self.tonnage + 500.0, self.mn, self.fe)
        self.assertFalse(v["all_pass"])
        t_check = next(c for c in v["checks"]
                       if c["constraint"].startswith("Total"))
        self.assertEqual(t_check["status"], "FAIL")

    def test_budget_constraint_in_verify(self):
        result = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0,
                                         max_cost_per_ton=1500.0)
        self.assertTrue(result["success"])
        v = verify_blend(result, 5000.0, 33.0, 20.0, max_cost_per_ton=1500.0)
        self.assertTrue(v["all_pass"], v)
        self.assertTrue(any(c["constraint"].startswith("Maximum cost")
                            for c in v["checks"]))

    def test_budget_violation_detected(self):
        result = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0,
                                         max_cost_per_ton=1500.0)
        v = verify_blend(result, 5000.0, 33.0, 20.0, max_cost_per_ton=900.0)
        self.assertFalse(v["all_pass"])
        c = next(x for x in v["checks"] if x["constraint"].startswith("Maximum cost"))
        self.assertEqual(c["status"], "FAIL")

    def test_failed_optimization_not_verifyable(self):
        failed = calculate_optimal_blend(self.stockpiles, 50000.0, self.mn,
                                         self.fe)
        self.assertFalse(failed["success"])
        v = verify_blend(failed, 50000.0, self.mn, self.fe)
        self.assertFalse(v["all_pass"])
        self.assertTrue(v["reason"])

    def test_infeasible_with_budget(self):
        result = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0,
                                         max_cost_per_ton=1.0)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "Infeasible")

    def test_explicit_budget_ignored_when_none(self):
        # Passing None must behave exactly like the default (no budget).
        base = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0)
        none_budget = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0,
                                              max_cost_per_ton=None)
        self.assertEqual(base["cost_per_ton"], none_budget["cost_per_ton"])

    def test_tiny_allocations_preserved_and_constraints_valid(self):
        # Regression: the old post-processing zeroed every allocation below
        # 0.5 t, silently corrupting total tonnage and the Mn/Fe/cost checks.
        # Tiny-but-real solver values must be preserved; only floating-point
        # noise below 1e-8 may be zeroed.
        stock = pd.DataFrame({
            "Stockpile_ID": ["S-LOW", "S-TINY-1", "S-TINY-2"],
            "Available_Tons": [6500.0, 0.05, 0.08],
            "Mn_Grade": [30.0, 60.0, 58.0],
            "Fe_Grade": [10.0, 5.0, 6.0],
            "Cost_per_Ton": [100.0, 101.0, 103.0],
        })
        target = 6500.13
        result = calculate_optimal_blend(stock, target, 20.0, 30.0)
        self.assertTrue(result["success"], result.get("message"))

        alloc = result["allocations"]
        tiny1 = alloc[np.where(result["result_df"]["Stockpile_ID"] == "S-TINY-1")[0]][0]
        tiny2 = alloc[np.where(result["result_df"]["Stockpile_ID"] == "S-TINY-2")[0]][0]
        # Both tiny allocations exist, are positive, but below the old 0.5-t
        # deletion threshold - they must NOT be discarded.
        self.assertGreater(tiny1, 0.0)
        self.assertGreater(tiny2, 0.0)
        self.assertLess(tiny1, 0.5)
        self.assertLess(tiny2, 0.5)
        self.assertAlmostEqual(tiny1, 0.05, places=5)
        self.assertAlmostEqual(tiny2, 0.08, places=5)

        # Total tonnage from the *exact* allocations equals the target.
        self.assertAlmostEqual(alloc.sum(), target, places=5)

        # Constraints must be validated against the actual solver result.
        v = verify_blend(result, target, 20.0, 30.0)
        self.assertTrue(v["all_pass"], v)

        # Display column is the only place rounding may occur; it must remain
        # consistent with the internal (exact) allocations.
        display = result["result_df"]["Allocated_Tons"].to_numpy()
        np.testing.assert_allclose(display, np.round(alloc, 2))

    def test_allocation_display_matches_internal(self):
        # For the standard demo blend, the displayed allocation (rounded to
        # 2 dp) must remain consistent with the internal solver allocation.
        result = calculate_optimal_blend(self.stockpiles, 5000.0, 33.0, 20.0)
        self.assertTrue(result["success"])
        alloc = result["allocations"]
        display = result["result_df"]["Allocated_Tons"].to_numpy()
        np.testing.assert_allclose(display, np.round(alloc, 2))
        self.assertAlmostEqual(alloc.sum(), 5000.0, places=5)


# ===========================================================================
# Module 6 - Upload workflow regression (sample_geological_data.csv)
# ===========================================================================

class TestUploadWorkflow(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(SAMPLE_GEO_CSV),
                        f"sample CSV missing: {SAMPLE_GEO_CSV}")

    def test_sample_csv_loads(self):
        with _open_sample_csv() as up:
            df = parse_upload(up, "sample_geological_data.csv")
        self.assertGreater(len(df), 0)

    def test_required_columns_detected(self):
        with _open_sample_csv() as up:
            df = parse_upload(up, "sample_geological_data.csv")
        for col in ["Borehole_ID", "X", "Y", "Depth", "Mn_Grade", "Fe_Grade"]:
            self.assertIn(col, df.columns, f"missing {col}")

    def test_validation_passes(self):
        with _open_sample_csv() as up:
            df = parse_upload(up, "sample_geological_data.csv")
        vg = validate_geological_data(df)
        self.assertTrue(vg["ok"], vg["errors"])
        self.assertEqual(len(vg["cleaned"]), len(df))

    def test_kriging_uses_uploaded_sample(self):
        with _open_sample_csv() as up:
            df = parse_upload(up, "sample_geological_data.csv")
        model = krige_uploaded_samples(df, 50, 3.0)
        self.assertGreater(len(model), 0)
        self.assertFalse(model.isna().any().any())
        self.assertTrue((model["Mn_Grade"] >= 0).all())
        self.assertTrue((model["Mn_Grade"] <= 100).all())
        self.assertTrue((model["Fe_Grade"] >= 0).all())
        self.assertTrue((model["Fe_Grade"] <= 100).all())

    def test_uploaded_grid_within_configured_limit(self):
        with _open_sample_csv() as up:
            df = parse_upload(up, "sample_geological_data.csv")
        model = krige_uploaded_samples(df, 50, 3.0)
        self.assertLessEqual(model["X"].nunique(), MAX_GRID_CELLS)
        self.assertLessEqual(model["Y"].nunique(), MAX_GRID_CELLS)


# ===========================================================================
# Module 7 - Horizontal grid size cap regression (wide / elongated extents)
# ===========================================================================

def _wide_strip_samples(span_x: float = 3200.0, span_y: float = 100.0) -> pd.DataFrame:
    """Synthetic sample grid over a highly unequal horizontal extent."""
    rng = np.random.default_rng(7)
    xs = np.linspace(0.0, span_x, 33)
    ys = np.linspace(0.0, span_y, 8)
    rows = []
    for x in xs:
        for y in ys:
            rows.append({
                "Borehole_ID": f"W-{len(rows):04d}",
                "X": x, "Y": y,
                "Depth": 15.0,
                "Mn_Grade": float(np.clip(28.0 + 0.004 * x + rng.normal(0, 1.5),
                                          15.0, 50.0)),
                "Fe_Grade": float(np.clip(20.0 - 0.002 * x + rng.normal(0, 1.0),
                                          10.0, 40.0)),
            })
    return pd.DataFrame(rows)


class TestGridCapRegression(unittest.TestCase):

    def test_wide_strip_never_exceeds_grid_limit(self):
        # Stress test: X span ~3200 m, Y span ~100 m. The old cell-size logic
        # clamped using min() per-axis so the long dimension could exceed
        # MAX_GRID_CELLS. The capped cell size must bound BOTH dimensions.
        samples = _wide_strip_samples(3200.0, 100.0)
        model = krige_uploaded_samples(samples, 50, 3.0)
        nx, ny = model["X"].nunique(), model["Y"].nunique()
        self.assertGreater(len(model), 0)
        self.assertFalse(model.isna().any().any())
        self.assertLessEqual(nx, MAX_GRID_CELLS, f"X axis: {nx} cells")
        self.assertLessEqual(ny, MAX_GRID_CELLS, f"Y axis: {ny} cells")

    def test_wide_strip_preserves_spatial_coverage(self):
        samples = _wide_strip_samples(3200.0, 100.0)
        model = krige_uploaded_samples(samples, 50, 3.0)
        # Cells are square: derive the cell size from the X spacing.
        ux = np.sort(model["X"].unique())
        cell = float(np.diff(ux).min())
        uy = np.sort(model["Y"].unique())
        # The model spans the full observed X range (within one cell).
        self.assertGreaterEqual(ux[-1] - ux[0], 3100.0)
        # Full spatial coverage: every block center cell (center +/- cell/2)
        # covers all synthetic sample coordinates - nothing is arbitrarily
        # dropped to fit the grid cap.
        self.assertLessEqual(ux[0] - cell / 2, float(samples["X"].min()))
        self.assertGreaterEqual(ux[-1] + cell / 2, float(samples["X"].max()))
        self.assertLessEqual(uy[0] - cell / 2, float(samples["Y"].min()))
        self.assertGreaterEqual(uy[-1] + cell / 2, float(samples["Y"].max()))

    def test_balanced_extent_keeps_block_cell_minimum(self):
        # A roughly square, small extent must keep the requested block_cell
        # (no unnecessary upscaling when the span is already small).
        samples = get_demonstration_samples(PIT)
        model = krige_uploaded_samples(samples, 50, 3.0, block_cell=25.0)
        nx, ny = model["X"].nunique(), model["Y"].nunique()
        self.assertLessEqual(nx, MAX_GRID_CELLS)
        self.assertLessEqual(ny, MAX_GRID_CELLS)
        self.assertGreater(nx, 1)
        self.assertGreater(ny, 1)


# ===========================================================================
# Module 8 - Bundled sample geological dataset: model quality
# ===========================================================================

def _sample_frame() -> pd.DataFrame:
    with _open_sample_csv() as up:
        return parse_upload(up, "sample_geological_data.csv")


class TestSampleModelQuality(unittest.TestCase):
    """The bundled sample must demonstrate the full geological workflow:
    enough coverage/depth variation for a meaningful 3D block model, valid
    grades, no NaNs, and a screening output that contains both classes."""

    def test_sample_is_substantial(self):
        df = _sample_frame()
        vg = validate_geological_data(df)
        self.assertTrue(vg["ok"], vg["errors"])
        self.assertGreaterEqual(len(vg["cleaned"]), 1000,
                                "sample too small to demonstrate 3D kriging")
        self.assertGreaterEqual(vg["cleaned"]["Borehole_ID"].nunique(), 100)
        self.assertGreaterEqual(vg["cleaned"]["Depth"].nunique(), 5,
                                "needs vertical (depth) variation")

    def test_sample_produces_meaningful_block_model(self):
        df = _sample_frame()
        model = krige_uploaded_samples(df, DEFAULT_DEPTH, DEFAULT_CUTOFF)
        self.assertGreaterEqual(len(model), 500,
                                "sample must yield a substantial block model")
        self.assertFalse(model.isna().any().any())
        self.assertTrue((model["Mn_Grade"] >= 0).all())
        self.assertTrue((model["Mn_Grade"] <= 100).all())
        self.assertTrue((model["Fe_Grade"] >= 0).all())
        self.assertTrue((model["Fe_Grade"] <= 100).all())
        # 3D model: several horizontal cells and several depth layers.
        self.assertGreater(model["X"].nunique(), 4)
        self.assertGreater(model["Y"].nunique(), 4)
        self.assertGreater(model["Z"].nunique(), 1)
        self.assertLessEqual(model["X"].nunique(), MAX_GRID_CELLS)
        self.assertLessEqual(model["Y"].nunique(), MAX_GRID_CELLS)

    def test_sample_screening_contains_both_classes(self):
        # Screening is not an on/off artifact: some blocks should be UNFC 111
        # and some UNVERIFIED, otherwise the criterion is degenerate.
        df = _sample_frame()
        model = krige_uploaded_samples(df, DEFAULT_DEPTH, DEFAULT_CUTOFF)
        n111 = int((model["UNFC_Code"] == "111").sum())
        n_unv = int((model["UNFC_Code"] == "UNVERIFIED").sum())
        self.assertGreater(n111, 0)
        self.assertGreater(n_unv, 0)


# ===========================================================================
# Module 9 - KPI / data-source state consistency (no UI-data mixing)
# ===========================================================================

def _kpi_value(at: AppTest, label: str) -> str | None:
    for md in at.markdown:
        value = getattr(md, "value", "") or ""
        if "kpi-label" in value and label in value:
            m = re.search(r"kpi-label\">" + re.escape(label) +
                          r"</div><div class=\"kpi-value\">([^<]+)<", value)
            if m:
                return m.group(1)
    return None


class TestGeoKPIStateConsistency(unittest.TestCase):
    """Regression: after running an uploaded geological dataset, every
    displayed KPI must come from the uploaded model - never a stale mix of
    the earlier demonstration state. (Issue: demo KPIs persisted while the
    uploaded model was active.)"""

    @classmethod
    def setUpClass(cls):
        cls.demo_df = get_kriged_block_model(DEFAULT_PIT, DEFAULT_DEPTH,
                                             DEFAULT_CUTOFF)

    def _app_with_uploaded_model(self) -> AppTest:
        at = AppTest.from_file("app.py", default_timeout=300)
        at.run()
        nav = next(r for r in at.radio if r.label == "Navigation")
        nav.set_value("Geospatial AI")
        at.run()
        src = next(r for r in at.radio
                   if r.label == "Geological Data Source")
        src.set_value("Upload Geological Data")
        at.run()
        up = next(f for f in at.file_uploader
                  if f.label.startswith("Upload geological"))
        with _open_sample_csv() as fh:
            up.set_value(("sample_geological_data.csv", fh.read(),
                          "text/csv"))
        at.run()
        run_btn = next(b for b in at.button
                       if b.label == "Run Kriging Model")
        run_btn.click()
        at.run()
        return at

    def test_upload_run_drives_all_kpis(self):
        at = self._app_with_uploaded_model()
        self.assertFalse(at.exception)
        expected = krige_uploaded_samples(
            _sample_frame(), DEFAULT_DEPTH, DEFAULT_CUTOFF)
        self.assertNotEqual(len(expected), len(self.demo_df))
        self.assertEqual(_kpi_value(at, "Total Blocks"),
                         f"{len(expected):,}")
        self.assertEqual(_kpi_value(at, "Avg Mn Grade"),
                         f"{expected['Mn_Grade'].mean():.2f}%")
        self.assertEqual(_kpi_value(at, "Avg Fe Grade"),
                         f"{expected['Fe_Grade'].mean():.2f}%")
        self.assertEqual(
            _kpi_value(at, "UNFC 111 Screening"),
            f"{(expected['UNFC_Code'] == '111').sum():,}")
        # Demonstrating data is NOT leaking into the KPI cards.
        self.assertNotEqual(_kpi_value(at, "Total Blocks"),
                            f"{len(self.demo_df):,}")

    def test_switching_back_to_demonstration_restores_demo_kpis(self):
        at = AppTest.from_file("app.py", default_timeout=300)
        at.run()
        nav = next(r for r in at.radio if r.label == "Navigation")
        nav.set_value("Geospatial AI")
        at.run()
        src = next(r for r in at.radio
                   if r.label == "Geological Data Source")
        src.set_value("Upload Geological Data")
        at.run()
        up = next(f for f in at.file_uploader
                  if f.label.startswith("Upload geological"))
        with _open_sample_csv() as fh:
            up.set_value(("sample_geological_data.csv", fh.read(),
                          "text/csv"))
        at.run()
        next(b for b in at.button if b.label == "Run Kriging Model").click()
        at.run()
        # Switch back to the demonstration dataset and rebuild.
        src = next(r for r in at.radio
                   if r.label == "Geological Data Source")
        src.set_value("Demonstration Dataset")
        at.run()
        next(b for b in at.button if b.label == "Run Kriging Model").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(_kpi_value(at, "Total Blocks"),
                         f"{len(self.demo_df):,}")


def run_silently() -> tuple[unittest.TextTestResult, str]:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    stream = io.StringIO()
    runner = unittest.TextTestRunner(verbosity=2, stream=stream)
    result = runner.run(suite)
    return result, stream.getvalue()


if __name__ == "__main__":
    result, output = run_silently()
    print(output)
    print("=" * 70)
    print(f"MnBrain - module tests: ran {result.testsRun} tests")
    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED")
    else:
        print(f"RESULT: {len(result.failures)} failures, "
              f"{len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)