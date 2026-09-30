from ctypes import WinDLL, get_last_error
from ctypes import c_ssize_t as LONG_PTR
from ctypes.wintypes import BOOL, BYTE, COLORREF, DWORD, HWND, INT

GWL_EXSTYLE = -20

WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000

WDA_EXCLUDEFROMCAPTURE = 0x00000011

LWA_COLORKEY = 0x00000001

user32 = WinDLL("user32", use_last_error=True)

GetWindowLongPtrW = user32.GetWindowLongPtrW
GetWindowLongPtrW.argtypes = HWND, INT
GetWindowLongPtrW.restype = LONG_PTR

SetWindowLongPtrW = user32.SetWindowLongPtrW
SetWindowLongPtrW.argtypes = HWND, INT, LONG_PTR
SetWindowLongPtrW.restype = LONG_PTR

SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
SetWindowDisplayAffinity.argtypes = HWND, DWORD
SetWindowDisplayAffinity.restype = BOOL

SetLayeredWindowAttributes = user32.SetLayeredWindowAttributes
SetLayeredWindowAttributes.argtypes = HWND, COLORREF, BYTE, DWORD
SetLayeredWindowAttributes.restype = BOOL


class WinAPI:
    @staticmethod
    def set_transparency(hwnd: int, color_rgb: tuple[int, int, int]) -> None:

        ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
        ex_style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex_style)

        r, g, b = color_rgb
        colorref = r | (g << 8) | (b << 16)
        SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)

    @staticmethod
    def exclude_from_capture(hwnd: int) -> None:

        success = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

        if not success:
            raise RuntimeError(
                f"Failed to set WDA_EXCLUDEFROMCAPTURE: {get_last_error()}"
            )
