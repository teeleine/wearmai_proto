# Generated by Django 5.1.2 on 2024-10-15 09:31

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_alter_run_options_remove_run_speed_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='run',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='runs', to='core.userprofile'),
        ),
    ]
