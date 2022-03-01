from .dist import map_faces, compare_faces
from .interpolate import interpolate
from .dynamic import dynamically_process, make_sequences
from .file import read_frame, write_faces
from .face_recognition import remove_feats

__all__ = ["map_faces", "compare_faces", "interpolate",
           "dynamically_process", "make_sequences", "read_frame", "write_faces", "remove_feats"]
