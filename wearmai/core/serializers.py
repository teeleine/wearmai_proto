from rest_framework import serializers
from .models import Run, UserProfile
from core.models import Run, UserProfile, ExerciseUnit
from services.exercise_summarisation.exercise_summary_service import ExerciseSummaryService

class RunSerializer(serializers.ModelSerializer):
    class Meta:
        model = Run
        fields = ['id', 'date']

class RunDetailSerializer(serializers.ModelSerializer):
    kilometers = serializers.SerializerMethodField()

    averages_across_runs = serializers.SerializerMethodField()

    class Meta:
        model = Run
        fields = ['id', 'date', 'kilometers', 'averages_across_runs']

    def get_averages_across_runs(self, obj):
        exercise_units = [e for e in ExerciseUnit.objects.filter(run=obj)]
        return ExerciseSummaryService(exercise_units).run(aggregate = True)

    def get_kilometers(self, obj):
        kilometers = {}
        for index, exercise_unit in enumerate(obj.exercise_units.all()):
            kilometers[f"kilometer_{index}"] = {
                'speed': exercise_unit.speed,
                'summary': ExerciseSummaryService([exercise_unit]).run(aggregate = True)
            }
        return kilometers


class UserProfileForLLM(serializers.ModelSerializer):
    user_summary = serializers.SerializerMethodField()
    class Meta:
        model = UserProfile
        fields = ['id', 'name', 'weight', 'height', 'user_summary']


    def get_user_summary(self, obj):
        exercise_units = [e for e in ExerciseUnit.objects.filter(run__in=obj.runs.all())]
        ai_user_profile = {
            'runs': {
                'aggregated_run_summary': ExerciseSummaryService(exercise_units).run(aggregate = True),
                'run_data': RunSerializer(Run.objects.filter(user=obj), many=True).data
            }
        }
        return ai_user_profile