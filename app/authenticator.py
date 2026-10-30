import pyotp
import qrcode
from PIL import Image, ImageTk
import json
from pathlib import Path

class AuthenticatorManager:
    def __init__(self, app_name: str = "FaceRegisterApp"):
        self.app_name = app_name

    def generate_secret(self) -> str:
        # Funkcja odpowiedzialna za wygenerowanie nowego klucza tajnego dla użytkownika.
        return pyotp.random_base32()

    def get_auth_uri(self, secret: str, username: str) -> str:
        # Funkcja odpowiedzialna za utworzenie URI dla aplikacji uwierzytelniającej.
        return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=self.app_name)

    def generate_qr_code(self, uri: str, size: int = 250) -> ImageTk.PhotoImage:
        # Funkcja odpowiedzialna za utworzenie obrazu kodu QR kompatybilnego z Tkinter.
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(uri)
        qr.make(fit=True)

        # Konwersja do RGB zapobiega problemom z rozmiarem obrazu w starszych wersjach PIL.
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize((size, size))
        return ImageTk.PhotoImage(img)

    def verify_code(self, secret: str, user_code: str) -> bool:
        # Funkcja odpowiedzialna za weryfikację kodu podanego przez użytkownika.
        if not user_code or not user_code.isdigit():
            return False

        totp = pyotp.TOTP(secret)
        return totp.verify(user_code)

    def save_secret(self, user_dir: Path, secret: str) -> bool:
        # Funkcja odpowiedzialna za zapisanie klucza tajnego użytkownika do pliku.
        try:
            # Upewniamy się, że katalog profilu użytkownika istnieje przed zapisem.
            user_dir.mkdir(parents=True, exist_ok=True)
            auth_file = user_dir / "auth_secret.json"
            with open(auth_file, "w", encoding="utf-8") as f:
                json.dump({"secret": secret}, f)
            return True
        except Exception:
            return False

    def load_secret(self, user_dir: Path) -> str | None:
        # Funkcja odpowiedzialna za odczytanie klucza tajnego użytkownika z pliku.
        try:
            auth_file = user_dir / "auth_secret.json"
            if not auth_file.exists():
                return None
            with open(auth_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("secret")
        except Exception:
            return None