import cv2

def cameraStart(processFrame):
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    vc = cv2.VideoCapture(0)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vc.set(cv2.CAP_PROP_FPS, 30)

    print(
        f"Camera: {int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{vc.get(cv2.CAP_PROP_FPS):.0f} FPS"
    )

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = cv2.flip(frame, 1)
        processFrame(frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27:
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == "__main__":
    cameraStart(lambda frame: cv2.imshow("preview", frame))
