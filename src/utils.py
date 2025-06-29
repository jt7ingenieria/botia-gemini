# src/utils.py
# Módulo para funciones de utilidad.
# Entradas: Varias.
# Salidas: Varias.

import logging
import smtplib
from email.mime.text import MIMEText

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_message(message):
    """
    Registra un mensaje en el log.
    """
    logger.info(message)

def send_notification(subject: str, message: str):
    """
    Envía una notificación por correo electrónico.
    Configuración de correo electrónico (ejemplo, reemplazar con valores reales o variables de entorno)
    """
    sender_email = "your_email@example.com"
    sender_password = "your_email_password"
    receiver_email = "recipient_email@example.com"
    smtp_server = "smtp.example.com"
    smtp_port = 587 # O 465 para SSL

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls() # Habilitar seguridad TLS
            server.login(sender_email, sender_password)
            server.send_message(msg)
        log_message(f"Notificación enviada: {subject}")
    except Exception as e:
        log_message(f"Error al enviar notificación: {e}")
