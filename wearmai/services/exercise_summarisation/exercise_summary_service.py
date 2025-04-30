from typing import List
from core.models import ExerciseUnit, Soleus, Pelvis, PelvisRightSide, PelvisLeftSide, TibialisAnteriorLeftSide, TibialisAnteriorRightSide, MedialGastrocnemiusLeftSide, MedialGastrocnemiusRightSide, LateralGastrocnemiusLeftSide, LateralGastrocnemiusRightSide
from core.models import SoleusLeftSide, SoleusRightSide, TibialisAnterior, MedialGastrocnemius, LateralGastrocnemius, Hip, Knee, Ankle, HipRightSide, HipLeftSide, KneeRightSide, KneeLeftSide, AnkleRightSide, AnkleLeftSide
from common.utils.stats import get_summary

BODY_PARTS_TO_COLS = {
        # Soleus: ['force_avg', 'force_std'],
        # TibialisAnterior: ['force_avg', 'force_std'],
        # MedialGastrocnemius: ['force_avg', 'force_std'],
        # LateralGastrocnemius: ['force_avg', 'force_std'],
        Hip: ['flexion_avg', 'adduction_avg', 'rotation_avg', 'flexion_std', 'adduction_std', 'rotation_std'],
        Knee: ['angle_avg', 'angle_std'],
        Ankle: ['subtalar_angle_avg', 'angle_avg', 'subtalar_angle_std', 'angle_std'],
        Pelvis: ['tilt_angle_avg', 'list_angle_avg', 'rotation_angle_avg', 'tilt_angle_std', 'list_angle_std', 'rotation_angle_std']

    }


def aggregate_summaries(summaries):
    aggregated_summary = {}
    # Calculate a rolling average of the columns
    count = 1
    for summary in summaries:
        for body_part, sides in summary.items():
            if body_part not in aggregated_summary:
                aggregated_summary[body_part] = sides
            else:
                for side, cols in sides.items():
                    if side not in aggregated_summary[body_part]:
                        aggregated_summary[body_part][side] = cols
                    else:
                        for col, values in cols.items():
                            if col not in aggregated_summary[body_part][side]:
                                aggregated_summary[body_part][side][col] = values
                            else:
                                for key, value in values.items():
                                    aggregated_summary[body_part][side][col][key] = ((aggregated_summary[body_part][side][col][key]*count) + value)/(count + 1)
                                    count += 1

    # Iterate through again and round the values
    percision = 4
    for body_part, sides in aggregated_summary.items():
        for side, cols in sides.items():
            for col, values in cols.items():
                for key, value in values.items():
                    aggregated_summary[body_part][side][col][key] = round(value, percision)
    return aggregated_summary

class ExerciseSummaryService():
    def __init__(self, exercise_units: List[ExerciseUnit]):
        self.exercise_units = exercise_units

    def run(self, aggregate = False):
        summaries = []
        for exercise_unit in self.exercise_units:
            gait_phases = exercise_unit.gait_phases.all()
            for body_part, cols in BODY_PARTS_TO_COLS.items():
                main_body_parts = body_part.objects.filter(gait_phase__in=gait_phases)
                body_part_name = body_part.__name__
                left_side_class_name = f"{body_part.__name__}LeftSide"
                right_side_class_name = f"{body_part.__name__}RightSide"
                # get the actual class from teh class name string
                left_side_class = globals()[left_side_class_name]
                right_side_class = globals()[right_side_class_name]

                # use the bodypart class name as the query parameter
                # Title case to snake case
                body_part_name = ''.join(['_' + i.lower() if i.isupper() else i for i in body_part_name]).lstrip('_')
        

                left_side = left_side_class.objects.filter(**{f"{body_part_name}__in": main_body_parts})
                right_side = right_side_class.objects.filter(**{f"{body_part_name}__in": main_body_parts})

                summary = {
                    body_part.__name__: {
                        left_side_class_name: {
                            col: get_summary([getattr(left, col) for left in left_side])
                            for col in cols
                        },
                        right_side_class_name: {
                            col: get_summary([getattr(right, col) for right in right_side])
                            for col in cols
                        }
                }}
                summaries.append(summary)

        if aggregate:
            return aggregate_summaries(summaries)
        return summaries