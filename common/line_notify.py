import requests

class LineNotify(object):
    @staticmethod
    def send_line_notify(notify_flg, token, message):
        if notify_flg:
            line_notify_api = 'https://notify-api.line.me/api/notify'
            headers = {'Authorization': f'Bearer {token}'}
            data = {'message': f'message: {message}'}
            requests.post(line_notify_api, headers = headers, data = data)
        else:
            pass
