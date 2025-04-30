from django.db import models
from openai import OpenAI
import os
from typing import Generator
from wearmai.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)




class Conversation(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    user_profile = models.ForeignKey("UserProfile", on_delete=models.CASCADE)

    def prompt_stream(self, system_prompt="You are a helpful assistant", model="gpt-4o", functions = []) -> Generator[str, None, None]:            
    
        messages = [
            {"role": "system", "content": system_prompt},
            *self.to_chat()
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto",
            stream=True  # Enable streaming
        )

        accumulated_arguments = ""
        func_name = ""
        function_call = None
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content
                function_call = delta.function_call

                # Check if there's a function call
                if function_call:
                    func_arguments = function_call.arguments
                    if(function_call.name):
                        func_name = function_call.name

                    # Accumulate function call arguments
                    if func_arguments:
                        accumulated_arguments += func_arguments

                if content:
                    yield {'message': content}

        # If the function name is present, the function call is complete
        if func_name != "":
            # Forward the function call details to the frontend
            full_function_call_definition = {
                'name': func_name,
                'arguments': accumulated_arguments
            }
            yield {'function_call': full_function_call_definition}

        yield {'completed': True}

    def to_chat(self):
        return [
            message.message
            for message in self.message_set.order_by("created_at")
        ]

class Message(models.Model):
    PARTICIPANTS = (("user", "User"), ("assistant", "Ai"), ("system", "System"))

    participant = models.CharField(max_length=255, choices=PARTICIPANTS)
    conversation = models.ForeignKey("Conversation", on_delete=models.CASCADE)
    message = models.JSONField(default = {})
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

# Define Soleus and SoleusSide with one-to-one relationships
class SoleusRightSide(models.Model):
    soleus = models.OneToOneField('Soleus', on_delete=models.CASCADE, related_name='right_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class SoleusLeftSide(models.Model):
    soleus = models.OneToOneField('Soleus', on_delete=models.CASCADE, related_name='left_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class Soleus(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='soleus')


# Define Tibialis Anterior and TibialisAnteriorSide with one-to-one relationships
class TibialisAnteriorRightSide(models.Model):
    tibialis_anterior = models.OneToOneField('TibialisAnterior', on_delete=models.CASCADE, related_name='right_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class TibialisAnteriorLeftSide(models.Model):
    tibialis_anterior = models.OneToOneField('TibialisAnterior', on_delete=models.CASCADE, related_name='left_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class TibialisAnterior(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='tibialis_anterior')


# Define Medial Gastrocnemius and MedialGastrocnemiusSide with one-to-one relationships
class MedialGastrocnemiusRightSide(models.Model):
    medial_gastrocnemius = models.OneToOneField('MedialGastrocnemius', on_delete=models.CASCADE, related_name='right_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class MedialGastrocnemiusLeftSide(models.Model):
    medial_gastrocnemius = models.OneToOneField('MedialGastrocnemius', on_delete=models.CASCADE, related_name='left_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class MedialGastrocnemius(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='medial_gastrocnemius')


# Define Lateral Gastrocnemius and LateralGastrocnemiusSide with one-to-one relationships
class LateralGastrocnemiusRightSide(models.Model):
    lateral_gastrocnemius = models.OneToOneField('LateralGastrocnemius', on_delete=models.CASCADE, related_name='right_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class LateralGastrocnemiusLeftSide(models.Model):
    lateral_gastrocnemius = models.OneToOneField('LateralGastrocnemius', on_delete=models.CASCADE, related_name='left_side')
    force_avg = models.FloatField(null=True)
    force_std = models.FloatField(null=True)


class LateralGastrocnemius(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='lateral_gastrocnemius')


# Define Hip and HipSide with one-to-one relationships
class HipRightSide(models.Model):
    hip = models.OneToOneField('Hip', on_delete=models.CASCADE, related_name='right_side')
    flexion_avg = models.FloatField(null=True)
    adduction_avg = models.FloatField(null=True)
    rotation_avg = models.FloatField(null=True)
    flexion_std = models.FloatField(null=True)
    adduction_std = models.FloatField(null=True)
    rotation_std = models.FloatField(null=True)


class HipLeftSide(models.Model):
    hip = models.OneToOneField('Hip', on_delete=models.CASCADE, related_name='left_side')
    flexion_avg = models.FloatField(null=True)
    adduction_avg = models.FloatField(null=True)
    rotation_avg = models.FloatField(null=True)
    flexion_std = models.FloatField(null=True)
    adduction_std = models.FloatField(null=True)
    rotation_std = models.FloatField(null=True)


class Hip(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='hip')


# Define Ankle, Knee, Pelvis with one-to-one relationships
class AnkleRightSide(models.Model):
    ankle = models.OneToOneField('Ankle', on_delete=models.CASCADE, related_name='right_side')
    subtalar_angle_avg = models.FloatField(null=True)
    angle_avg = models.FloatField(null=True)
    subtalar_angle_std = models.FloatField(null=True)
    angle_std = models.FloatField(null=True)


class AnkleLeftSide(models.Model):
    ankle = models.OneToOneField('Ankle', on_delete=models.CASCADE, related_name='left_side')
    subtalar_angle_avg = models.FloatField(null=True)
    angle_avg = models.FloatField(null=True)
    subtalar_angle_std = models.FloatField(null=True)
    angle_std = models.FloatField(null=True)


class Ankle(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='ankle')


class KneeRightSide(models.Model):
    knee = models.OneToOneField('Knee', on_delete=models.CASCADE, related_name='right_side')
    angle_avg = models.FloatField(null=True)
    angle_std = models.FloatField(null=True)


class KneeLeftSide(models.Model):
    knee = models.OneToOneField('Knee', on_delete=models.CASCADE, related_name='left_side')
    angle_avg = models.FloatField(null=True)
    angle_std = models.FloatField(null=True)


class Knee(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='knee')


class PelvisRightSide(models.Model):
    pelvis = models.OneToOneField('Pelvis', on_delete=models.CASCADE, related_name='right_side')
    tilt_angle_avg = models.FloatField(null=True)
    list_angle_avg = models.FloatField(null=True)
    rotation_angle_avg = models.FloatField(null=True)
    tilt_angle_std = models.FloatField(null=True)
    list_angle_std = models.FloatField(null=True)
    rotation_angle_std = models.FloatField(null=True)


class PelvisLeftSide(models.Model):
    pelvis = models.OneToOneField('Pelvis', on_delete=models.CASCADE, related_name='left_side')
    tilt_angle_avg = models.FloatField(null=True)
    list_angle_avg = models.FloatField(null=True)
    rotation_angle_avg = models.FloatField(null=True)
    tilt_angle_std = models.FloatField(null=True)
    list_angle_std = models.FloatField(null=True)
    rotation_angle_std = models.FloatField(null=True)


class Pelvis(models.Model):
    gait_phase = models.OneToOneField('GaitPhase', on_delete=models.CASCADE, related_name='pelvis')


# Define GaitPhase model
class GaitPhase(models.Model):
    exercise_unit = models.ForeignKey('ExerciseUnit', on_delete=models.CASCADE, related_name='gait_phases')
    phase = models.FloatField(null=True)


# Define ExerciseUnit and associated models
class ExerciseUnit(models.Model):
    run = models.ForeignKey('Run', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    walk = models.ForeignKey('Walk', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    jump = models.ForeignKey('Jump', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    squat = models.ForeignKey('Squat', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    land = models.ForeignKey('Land', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    lunge = models.ForeignKey('Lunge', on_delete=models.CASCADE, related_name='exercise_units', null=True, blank=True)
    speed = models.FloatField(null=True)
    # unit_count = models.FloatField(null=True)


# Define User and exercise-related models
class UserProfile(models.Model):
    name = models.CharField(max_length=255)
    height = models.FloatField(null=True)
    weight = models.FloatField(null=True)


class Run(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='runs')
    date = models.DateField()
    # duration = models.FloatField(null=True)

    class Meta:
        ordering = ['date']


class Walk(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='user_walks')
    date = models.DateField()
    # duration = models.FloatField(null=True)
    speed = models.FloatField(null=True)


class Jump(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='user_jumps')
    date = models.DateField()
    # duration = models.FloatField(null=True)


class Squat(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='user_squats')
    date = models.DateField()
    # duration = models.FloatField(null=True)


class Land(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='user_lands')
    date = models.DateField()
    # duration = models.FloatField(null=True)


class Lunge(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='user_lunges')
    date = models.DateField()
   
