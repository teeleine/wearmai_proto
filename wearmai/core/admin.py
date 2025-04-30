import nested_admin
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from collections import defaultdict

from .models import (
    UserProfile,
    Run, Walk, Jump, Squat, Land, Lunge,
    ExerciseUnit, GaitPhase,
    Soleus, SoleusRightSide, SoleusLeftSide,
    TibialisAnterior, TibialisAnteriorRightSide, TibialisAnteriorLeftSide,
    MedialGastrocnemius, MedialGastrocnemiusRightSide, MedialGastrocnemiusLeftSide,
    LateralGastrocnemius, LateralGastrocnemiusRightSide, LateralGastrocnemiusLeftSide,
    Hip, HipRightSide, HipLeftSide,
    Ankle, AnkleRightSide, AnkleLeftSide,
    Knee, KneeRightSide, KneeLeftSide,
    Pelvis, PelvisRightSide, PelvisLeftSide,
)

# ------------------------------
# Inline Classes for GaitPhase within ExerciseUnit
# ------------------------------

class GaitPhaseTabularInline(nested_admin.NestedTabularInline):
    """
    Tabular Inline for displaying GaitPhase data in a CSV-like table.
    """
    model = GaitPhase
    extra = 0
    ordering = ['phase']
    readonly_fields = (
        'phase',
        'ankle_angle_left',
        'ankle_angle_right',
        'hip_flexion_left',
        'hip_flexion_right',
        'knee_angle_left',
        'knee_angle_right',
        'pelvis_tilt_angle_left',
        'pelvis_tilt_angle_right',
        'soleus_left_force_avg',
        'soleus_right_force_avg',
        'soleus_left_force_std',
        'soleus_right_force_std',
        'tibialis_anterior_left_force_avg',
        'tibialis_anterior_right_force_avg',
        'tibialis_anterior_left_force_std',
        'tibialis_anterior_right_force_std',
        'medial_gastrocnemius_left_force_avg',
        'medial_gastrocnemius_right_force_avg',
        'medial_gastrocnemius_left_force_std',
        'medial_gastrocnemius_right_force_std',
        'lateral_gastrocnemius_left_force_avg',
        'lateral_gastrocnemius_right_force_avg',
        'lateral_gastrocnemius_left_force_std',
        'lateral_gastrocnemius_right_force_std'
    )
    fields = readonly_fields

    @admin.display(description="Ankle Angle Left")
    def ankle_angle_left(self, obj):
        if obj.ankle and hasattr(obj.ankle, 'left_side'):
            return obj.ankle.left_side.angle_avg
        return None

    @admin.display(description="Ankle Angle Right")
    def ankle_angle_right(self, obj):
        if obj.ankle and hasattr(obj.ankle, 'right_side'):
            return obj.ankle.right_side.angle_avg
        return None

    @admin.display(description="Hip Flexion Left")
    def hip_flexion_left(self, obj):
        if obj.hip and hasattr(obj.hip, 'left_side'):
            return obj.hip.left_side.flexion_avg
        return None

    @admin.display(description="Hip Flexion Right")
    def hip_flexion_right(self, obj):
        if obj.hip and hasattr(obj.hip, 'right_side'):
            return obj.hip.right_side.flexion_avg
        return None

    @admin.display(description="Knee Angle Left")
    def knee_angle_left(self, obj):
        if obj.knee and hasattr(obj.knee, 'left_side'):
            return obj.knee.left_side.angle_avg
        return None

    @admin.display(description="Knee Angle Right")
    def knee_angle_right(self, obj):
        if obj.knee and hasattr(obj.knee, 'right_side'):
            return obj.knee.right_side.angle_avg
        return None

    @admin.display(description="Pelvis Tilt Angle Left")
    def pelvis_tilt_angle_left(self, obj):
        if obj.pelvis and hasattr(obj.pelvis, 'left_side'):
            return obj.pelvis.left_side.tilt_angle_avg
        return None

    @admin.display(description="Pelvis Tilt Angle Right")
    def pelvis_tilt_angle_right(self, obj):
        if obj.pelvis and hasattr(obj.pelvis, 'right_side'):
            return obj.pelvis.right_side.tilt_angle_avg
        return None

    @admin.display(description="Soleus Left Force Avg")
    def soleus_left_force_avg(self, obj):
        if obj.soleus and hasattr(obj.soleus, 'left_side'):
            return obj.soleus.left_side.force_avg
        return None

    @admin.display(description="Soleus Right Force Avg")
    def soleus_right_force_avg(self, obj):
        if obj.soleus and hasattr(obj.soleus, 'right_side'):
            return obj.soleus.right_side.force_avg
        return None

    @admin.display(description="Soleus Left Force Std")
    def soleus_left_force_std(self, obj):
        if obj.soleus and hasattr(obj.soleus, 'left_side'):
            return obj.soleus.left_side.force_std
        return None

    @admin.display(description="Soleus Right Force Std")
    def soleus_right_force_std(self, obj):
        if obj.soleus and hasattr(obj.soleus, 'right_side'):
            return obj.soleus.right_side.force_std
        return None

    @admin.display(description="Tibialis Anterior Left Force Avg")
    def tibialis_anterior_left_force_avg(self, obj):
        if obj.tibialis_anterior and hasattr(obj.tibialis_anterior, 'left_side'):
            return obj.tibialis_anterior.left_side.force_avg
        return None

    @admin.display(description="Tibialis Anterior Right Force Avg")
    def tibialis_anterior_right_force_avg(self, obj):
        if obj.tibialis_anterior and hasattr(obj.tibialis_anterior, 'right_side'):
            return obj.tibialis_anterior.right_side.force_avg
        return None

    @admin.display(description="Tibialis Anterior Left Force Std")
    def tibialis_anterior_left_force_std(self, obj):
        if obj.tibialis_anterior and hasattr(obj.tibialis_anterior, 'left_side'):
            return obj.tibialis_anterior.left_side.force_std
        return None

    @admin.display(description="Tibialis Anterior Right Force Std")
    def tibialis_anterior_right_force_std(self, obj):
        if obj.tibialis_anterior and hasattr(obj.tibialis_anterior, 'right_side'):
            return obj.tibialis_anterior.right_side.force_std
        return None

    @admin.display(description="Medial Gastrocnemius Left Force Avg")
    def medial_gastrocnemius_left_force_avg(self, obj):
        if obj.medial_gastrocnemius and hasattr(obj.medial_gastrocnemius, 'left_side'):
            return obj.medial_gastrocnemius.left_side.force_avg
        return None

    @admin.display(description="Medial Gastrocnemius Right Force Avg")
    def medial_gastrocnemius_right_force_avg(self, obj):
        if obj.medial_gastrocnemius and hasattr(obj.medial_gastrocnemius, 'right_side'):
            return obj.medial_gastrocnemius.right_side.force_avg
        return None

    @admin.display(description="Medial Gastrocnemius Left Force Std")
    def medial_gastrocnemius_left_force_std(self, obj):
        if obj.medial_gastrocnemius and hasattr(obj.medial_gastrocnemius, 'left_side'):
            return obj.medial_gastrocnemius.left_side.force_std
        return None

    @admin.display(description="Medial Gastrocnemius Right Force Std")
    def medial_gastrocnemius_right_force_std(self, obj):
        if obj.medial_gastrocnemius and hasattr(obj.medial_gastrocnemius, 'right_side'):
            return obj.medial_gastrocnemius.right_side.force_std
        return None

    @admin.display(description="Lateral Gastrocnemius Left Force Avg")
    def lateral_gastrocnemius_left_force_avg(self, obj):
        if obj.lateral_gastrocnemius and hasattr(obj.lateral_gastrocnemius, 'left_side'):
            return obj.lateral_gastrocnemius.left_side.force_avg
        return None

    @admin.display(description="Lateral Gastrocnemius Right Force Avg")
    def lateral_gastrocnemius_right_force_avg(self, obj):
        if obj.lateral_gastrocnemius and hasattr(obj.lateral_gastrocnemius, 'right_side'):
            return obj.lateral_gastrocnemius.right_side.force_avg
        return None

    @admin.display(description="Lateral Gastrocnemius Left Force Std")
    def lateral_gastrocnemius_left_force_std(self, obj):
        if obj.lateral_gastrocnemius and hasattr(obj.lateral_gastrocnemius, 'left_side'):
            return obj.lateral_gastrocnemius.left_side.force_std
        return None

    @admin.display(description="Lateral Gastrocnemius Right Force Std")
    def lateral_gastrocnemius_right_force_std(self, obj):
        if obj.lateral_gastrocnemius and hasattr(obj.lateral_gastrocnemius, 'right_side'):
            return obj.lateral_gastrocnemius.right_side.force_std
        return None

# ------------------------------
# Inline Classes for ExerciseUnit
# ------------------------------

class ExerciseUnitInline(nested_admin.NestedStackedInline):
    """
    Inline for ExerciseUnit within each Exercise (Run, Walk, etc.).
    """
    model = ExerciseUnit
    extra = 0
    inlines = [GaitPhaseTabularInline]
    verbose_name = "Exercise Unit"
    verbose_name_plural = "Exercise Units"

# ------------------------------
# Separate Admin Classes for Each Exercise Type
# ------------------------------

class RunAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Run with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date')
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Run, RunAdmin)

class WalkAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Walk with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date')
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Walk, WalkAdmin)

class JumpAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Jump with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date',)
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Jump, JumpAdmin)

class SquatAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Squat with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date',)
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Squat, SquatAdmin)

class LandAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Land with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date',)
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Land, LandAdmin)

class LungeAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for Lunge with ExerciseUnitInline.
    """
    inlines = [ExerciseUnitInline]
    list_display = ('id', 'user', 'date',)
    list_filter = ('date', 'user',)
    search_fields = ('user__name',)

admin.site.register(Lunge, LungeAdmin)

# ------------------------------
# UserProfileAdmin with Exercises Grouped by Date
# ------------------------------

@admin.register(UserProfile)
class UserProfileAdmin(nested_admin.NestedModelAdmin):
    """
    Admin interface for UserProfile with exercises grouped by date and linked.
    """
    list_display = ('name', 'height', 'weight',)
    search_fields = ('name',)
    readonly_fields = ('exercises_by_date',)
    inlines = []

    def exercises_by_date(self, obj):
        """
        Displays a list of exercises grouped by date with links to each exercise's admin page.
        """
        # Collect all exercises related to the user
        exercises = []
        exercise_models = [Run, Walk, Jump, Squat, Land, Lunge]
        for model in exercise_models:
            exercises.extend(model.objects.filter(user=obj))

        # Group exercises by date
        exercises_grouped = defaultdict(list)
        for exercise in exercises:
            exercises_grouped[exercise.date].append(exercise)

        # Sort the dates
        sorted_dates = sorted(exercises_grouped.keys())

        # Build the HTML content
        html = ""
        for date in sorted_dates:
            html += f"<h3>{date}</h3><ul>"
            for exercise in exercises_grouped[date]:
                # Get the app_label and model_name dynamically
                app_label = exercise._meta.app_label
                model_name = exercise._meta.model_name

                # Reverse the admin change URL
                url = reverse(f'admin:{app_label}_{model_name}_change', args=(exercise.id,))
                # Display as "ExerciseType ID" e.g., "Run 1"
                display_name = f"{exercise.__class__.__name__} {exercise.id}"
                html += f'<li><a href="{url}">{display_name}</a></li>'
            html += "</ul>"

        return format_html(html)

    exercises_by_date.short_description = "Exercises by Date"
