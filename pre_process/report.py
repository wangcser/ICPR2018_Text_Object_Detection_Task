import os
import smtplib  # 发送邮件
import pre_process.config as cfg
from pre_process import timer

from email import encoders  # 附件编码
from email.mime.base import MIMEBase  # 附件标记
from email.mime.multipart import MIMEMultipart  # 多类型邮件容器
from email.mime.text import MIMEText  # 纯文本类型邮件


def send_email(attch_text=' ', send_to_addr=cfg.TO_EMAIL, file_path=None):
    """
    name: 发送邮件模块
    func: 调用该模块向指定地址发送邮件，已设定默认正文，默认收件人和默认附件
    send_email(attch_text='', send_to_addr='', file_path='')

    """

    # email 地址与用户口令
    from_addr = cfg.FROM_EMAIL
    password = cfg.FROM_EMAIL_TOKEN

    # 收件人地址
    to_addr = send_to_addr  # 默认收件地址,默认抄送一份到该地址

    # 构建一个支持附件的邮件容器
    msg = MIMEMultipart()

    # 构造当前时间戳，添加到邮件的主题中
    t = timer.timer()
    time_stamp = t.current_time()

    subject = '[' + time_stamp + '] ' + cfg.SUBJECT

    # 填写邮件头信息
    msg["Subject"] = subject  # 邮件主题
    msg["From"] = from_addr
    msg["To"] = to_addr

    # 填写邮件正文信息
    # 默认正文内容
    header_text = cfg.CONTENT

    # 希望 attch_text
    mime_text = MIMEText(header_text + attch_text, 'plain', 'utf-8')  # 实例化一个文本邮件对象
    msg.attach(mime_text)

    if not file_path is None:
        # 添加邮件附件信息
        with open(file_path, 'rb') as f:
            # 设置附件的MIME和文件名
            dir_path = os.path.dirname(file_path)
            file_name = file_path[len(dir_path):]  # 获取文件名
            mime = MIMEBase('database', 'db', filename=file_name)
            # 加上必要的头部信息
            mime.add_header('Content-Disposition', 'attachment', filename=file_name)
            mime.add_header('Content-ID', '<0>')
            mime.add_header('X-Attachment-Id', '0')
            # 读取附件内容
            mime.set_payload(f.read())
            # 用Base64编码
            encoders.encode_base64(mime)
            # 添加到MIMEMultipart
            msg.attach(mime)
            f.close()

    # 发送邮件
    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        # server.set_debuglevel(1) # 输出所有交互信息
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, msg.as_string())
        print('mail send.')
        server.quit()
    except:
        pass


if __name__ == '__main__':

    content = 'your content to send.'
    send_email(content)
