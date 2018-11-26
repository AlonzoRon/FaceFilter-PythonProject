import cv2
import face_recognition


class FaceFinder:

    def __init__(self, image, scaling=1):
        self.image = image
        self.scaling = 1 / scaling
        # scaling used to speed up facefinding

    def facelocation_size(self):
        face_loc = face_recognition.face_locations(self.image)
        face_location_size = []

        for top, right, bottom, left in face_loc:
            top *= self.scaling
            right *= self.scaling
            bottom *= self.scaling
            left *= self.scaling

            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)
            face_width = int(right - left)

            face_location_size.append([(center_x, center_y), face_width])
            # index 0 is face center, index 1 is face width (for each person)

        return face_location_size
        # Face size returned is width of the face (diameter)

    def bottomlip_location(self):
        face_landmarks = face_recognition.face_landmarks(self.image)

        bottom_lip_location = []

        for person in face_landmarks:
            bottom_lip = person['bottom_lip'][3]
            bottom_lipx, bottom_lipy = bottom_lip
            bottom_lipx *= self.scaling
            bottom_lipy *= self.scaling

            bottom_lip_location.append([int(bottom_lipx), int(bottom_lipy)])

        return bottom_lip_location

    def eyes_location(self):
        
        face_landmarks = face_recognition.face_landmarks(self.image)
        eyes_location = []

        for person in face_landmarks:
            right_eye = person['right_eye'][2]
            left_eye = person['left_eye'][2]

            right_eye_x, right_eye_y = right_eye
            left_eye_x, left_eye_y = left_eye

            right_eye_x *= self.scaling
            right_eye_y *= self.scaling
            left_eye_x *= self.scaling
            left_eye_y *= self.scaling

            right_eye_x = int(right_eye_x)
            right_eye_y = int(right_eye_y)
            left_eye_x = int(left_eye_x)
            left_eye_y = int(left_eye_y)

            eyes_location.append([[right_eye_x, right_eye_y],
                                  [left_eye_x, left_eye_y]])

        return eyes_location


class Overlayer:

    def __init__(self, image, decorator, x_center, y_center, diameter,
                 offset_x=0, offset_y=0):
        self.decorator = decorator
        self.x_center = x_center
        self.y_center = y_center
        self.diameter = int(diameter)
        self.image = image
        self.offset_y = int(-1 * offset_y)
        # offset y has to be negative to move image down
        self.offset_x = int(offset_x)

    def resizer(self, new_width):
        d_h, d_w, d_c = self.decorator.shape
        ratio = new_width / float(d_w)
        new_dimensions = (new_width, int(d_h * ratio))
        # line above calculates the ratio of new resized image

        resized_image = cv2.resize(self.decorator, new_dimensions, cv2.INTER_AREA) 
        return resized_image

    def overlay(self):
        # overlay is resized version of original png file
        overlay = self.resizer(new_width=self.diameter)
        overlay_w, overlay_h, overlay_c = overlay.shape
        image_h, image_w, image_c = self.image.shape

        radius = int(self.diameter / 2)
        origin_x = self.x_center - radius
        origin_y = self.y_center - radius

        # algorithm below is used to replace pixels from orig image
        for i in range(0, overlay_w):
            for j in range(0, overlay_h):

                overlay_location_x = origin_x + j + self.offset_x
                overlay_location_y = origin_y + i + self.offset_y

                if overlay_location_x >= image_w or overlay_location_y >= image_h:
                    continue
                if overlay_location_x < 0 or overlay_location_y < 0:
                    continue
                    # these lines of code above prevent crashing when index is out of bounds

                if overlay[i, j][3] != 0:  # transparency detect
                    self.image[overlay_location_y, overlay_location_x] = overlay[i, j]

        # overlay function edits sent image, no need to return value


class SnapChat:
    def __init__(self, scale=1):
        self.scale = scale
        # scale will define speed of snapchat face recognition
        # scale 1 = full size, scale 0.5 = half size
        # decorators will be more accurate for a higher value

    def mainmenu(self):
        print("""SnapChat Filter Thingy in Python:
Type in and enter corresponding number for corresponding filter:

    0 - Walter White-esque glasses, beard, and fedora
    1 - Googley Eyes
    2 - Flower Crown
    9 - Exit Program""")

        possible_choices = [0, 1, 2, 9]

        action = ''
        while action not in possible_choices: 
            action = int(input("Insert what you want: "))
            if action not in possible_choices:
                print("you selected an invalid number")

        if action == 9:
            exit()

        return action

    def main(self):
        webcam = cv2.VideoCapture(0)
        decision = self.mainmenu()

        # Decorators:
        glasses_raw = cv2.imread("glass2.png", -1)
        beard_raw = cv2.imread("beardy.png", -1)
        fedora_raw = cv2.imread("fedora.png", -1)
        googley_right_raw = cv2.imread("googley_right.png", -1)
        googley_left_raw = cv2.imread("googley_left.png", -1)
        tongue_raw = cv2.imread("tongue.png", -1)
        flowercrown_raw = cv2.imread("flowercrown.png", -1)
        horn_raw = cv2.imread("horn.png", -1)

        while True:
            ret, image = webcam.read()

            small_image = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale)

            # elements below are lists of lists of coordinates for each person
            faces = FaceFinder(small_image, self.scale)
            faces_locations_sizes = faces.facelocation_size()
            bottomlip_location = faces.bottomlip_location()
            eyes = faces.eyes_location()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            # convert image to BGRA for transparency check

            no_of_people = len(faces_locations_sizes)

            # gets the coordinates of landmarks for each person
            for i in range(no_of_people):
                face_center = faces_locations_sizes[i][0]  # in x, y tuple
                face_size = faces_locations_sizes[i][1]  # diameter of face (r-l)
                right_eye = eyes[i][0]  # in x, y list
                left_eye = eyes[i][1]  # in x, y list
                bottom_lip = bottomlip_location[i]  # in x, y list

                # get respective x, y coordinates
                fc_x, fc_y = face_center
                re_x, re_y = right_eye
                le_x, le_y = left_eye
                bl_x, bl_y = bottom_lip

                # create Overlayer objects (format below)
                # Overlayer(image, decorator, x_center, y_center, diameter,
                #           offset_x=0, offset_y=0): 
                # these objects let me manually adjust offset values

                # Walter White objects:
                glasses = Overlayer(image, glasses_raw, fc_x, fc_y, face_size, 0, -20)
                fedora = Overlayer(image, fedora_raw, fc_x, fc_y, face_size*0.9, 0, (face_size/2)+20)
                beard = Overlayer(image, beard_raw, bl_x, bl_y, face_size*0.7, 0, 0)
                # Googley Eyes objects:
                googley_right = Overlayer(image, googley_right_raw, re_x, re_y, face_size/3)
                googley_left = Overlayer(image, googley_left_raw, le_x, le_y, face_size/3)
                tongue = Overlayer(image, tongue_raw, bl_x, bl_y, face_size*0.5, face_size/10, -face_size/7)
                horn = Overlayer(image, horn_raw, fc_x, fc_y, face_size, 0, face_size/2)
                # Flower Crown objects:
                flowercrown = Overlayer(image, flowercrown_raw, fc_x, fc_y, face_size*1.3, 0, face_size/2)

                if decision == 0:  # Walter White
                    glasses.overlay()
                    fedora.overlay()
                    beard.overlay()

                if decision == 1:  # Googley Eyes
                    googley_right.overlay()
                    googley_left.overlay()
                    tongue.overlay()
                    horn.overlay()

                if decision == 2:  # Flower Crown
                    flowercrown.overlay()

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            # convert image back to BGR for viewing

            cv2.imshow("Snapchat (q for main menu)", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print()
        print('*************************************************************')

        webcam.release()
        cv2.destroyAllWindows()
        self.main()


snapchat = SnapChat(0.3)
snapchat.main()
