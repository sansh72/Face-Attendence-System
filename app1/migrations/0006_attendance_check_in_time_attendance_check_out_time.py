# Generated by Django 5.0.1 on 2024-07-10 08:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0005_attendance_student_delete_uploadedimage_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='attendance',
            name='timeslot',
            field=models.DateTimeField(blank=True, null=True),
        ),
        
    ]
