from django.contrib.auth.models import User
from django.http import HttpResponse

def create_admin(request):
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'insurepredictai', 'epoch')
        return HttpResponse("Superuser created.")
    else:
        return HttpResponse("Superuser already exists.")
