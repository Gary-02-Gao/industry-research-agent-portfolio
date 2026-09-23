#!/usr/bin/env python3
"""macOS local project launcher; only loopback listeners and project-owned processes."""
from __future__ import annotations
import argparse,fcntl,importlib.util,json,os,re,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
from urllib.parse import quote,unquote,urlsplit
from urllib.request import urlopen
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parent.parent
CONFIG=json.loads((ROOT/'.launcher/config.json').read_text())
RUNTIME=ROOT/'.runtime'
RUNTIME.mkdir(exist_ok=True)

def browse(url):
    if os.environ.get('PROJECT_NO_BROWSER')!='1':subprocess.run(['/usr/bin/open',url],check=True)

def free_port(preferred):
    for port in range(preferred,preferred+20):
        with socket.socket() as s:
            try:s.bind(('127.0.0.1',port));return port
            except OSError:pass
    raise RuntimeError(f'{preferred} 起的 20 个端口均被占用，请修改 .launcher/config.json 中的 port。')

def check_environment():
    assert sys.prefix!=sys.base_prefix,'没有使用项目隔离环境。'
    assert sys.version_info >= (3,10),'需要 Python 3.10 或更新版本。'
    print(f'项目：{ROOT.name}\nPython：{sys.version.split()[0]}\n隔离环境：{sys.prefix}',flush=True)
    for module in CONFIG.get('modules',[]):
        if importlib.util.find_spec(module) is None:raise RuntimeError(f'缺少依赖 {module}，请按 本地运行说明.md 安装。')
    if CONFIG['mode']=='docker':
        base=docker_command()
        subprocess.run(base+['config','--quiet'],cwd=ROOT/'project',check=True)
        print('Compose 配置有效；模型凭据在 project/.env 中单独填写。',flush=True)
    else:print('本地启动依赖检查通过。',flush=True)

def docker_command():
    binary=shutil.which('docker')
    if not binary:raise RuntimeError('未找到 Docker，请安装 Docker Desktop。')
    return [binary,'compose','--project-name','industry-research-curated','--env-file',str(ROOT/'project/.env'),'-f',str(ROOT/'project/docker-compose.yml')]

def docker_ready():
    binary=docker_command()[0]
    def ready():
        try:return subprocess.run([binary,'info'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8).returncode==0
        except subprocess.TimeoutExpired:return False
    if ready():return
    print('正在启动 Docker Desktop…',flush=True)
    subprocess.run(['/usr/bin/open','-a','Docker'],check=True)
    until=time.monotonic()+180
    while time.monotonic()<until:
        if ready():return
        time.sleep(2)
    raise RuntimeError('Docker Desktop 尚未就绪，请检查 Docker 窗口后重新双击。')

def read_public_setting(name,default):
    # Do not execute the environment file and do not print secrets.
    for line in (ROOT/'project/.env').read_text().splitlines():
        if line.strip().startswith(name+'='):
            return line.strip().split('=',1)[1].strip().strip('\"\'') or default
    return default

def run_docker(stop=False):
    docker_ready();base=docker_command()
    if stop:
        subprocess.run(base+['stop'],check=True,cwd=ROOT/'project')
        print('本项目容器已停止，数据库和上传资料保留。',flush=True);return
    check_environment()
    print('启动完整前后端及独立数据库容器；首次构建需要一些时间。',flush=True)
    with (RUNTIME/'docker-start.log').open('w') as log:
        proc=subprocess.Popen(base+['up','-d','--build','--wait','--wait-timeout','180'],cwd=ROOT/'project',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            result=proc.wait()
        except KeyboardInterrupt:
            proc.terminate();proc.wait(timeout=15);raise
    if result:raise RuntimeError('服务未全部启动，请查看 .runtime/docker-start.log；脚本没有删除任何数据卷。')
    port=read_public_setting('FRONTEND_PORT','15173')
    url=f'http://127.0.0.1:{port}/'
    with urlopen(url,timeout=10) as response:assert response.status==200
    (RUNTIME/'url.txt').write_text(url)
    print(f'\n前端：{url}\n后端：http://127.0.0.1:{read_public_setting("BACKEND_PORT","18000")}/docs',flush=True)
    key=read_public_setting('JOY_AGENT_API_KEY','')
    if not key or key.startswith('replace-'):
        print('模型服务尚未配置：可先打开界面、注册和登录；实时问答与向量化需填写 project/.env 后重新启动。',flush=True)
    print('关闭此终端不会停止 Docker 服务；需要停止时双击“停止项目.command”。',flush=True)
    browse(url)

class PortfolioHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
    def send_head(self):
        path=unquote(urlsplit(self.path).path)
        if any(x.startswith('.') for x in Path(path).parts if x not in {'/',''}):
            self.send_error(403);return None
        target=Path(self.translate_path(self.path)).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            self.send_error(404);return None
        self.byte_range=None
        value=self.headers.get('Range')
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            size=target.stat().st_size
            if not match or not any(match.groups()):self.send_error(416);return None
            left,right=match.groups()
            if left:start=int(left);end=min(int(right) if right else size-1,size-1)
            else:start=max(0,size-int(right));end=size-1
            if start>=size or end<start:self.send_error(416);return None
            f=target.open('rb');f.seek(start);self.byte_range=end-start+1
            self.send_response(206);self.send_header('Content-type',self.guess_type(str(target)))
            self.send_header('Content-Length',str(self.byte_range));self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.send_header('Accept-Ranges','bytes');self.end_headers();return f
        return super().send_head()
    def copyfile(self,source,outputfile):
        if self.byte_range is None:return super().copyfile(source,outputfile)
        remaining=self.byte_range
        while remaining:
            part=source.read(min(64*1024,remaining))
            if not part:break
            outputfile.write(part);remaining-=len(part)

def serve_portfolio(port):
    server=ThreadingHTTPServer(('127.0.0.1',port),PortfolioHandler)
    try:server.serve_forever()
    finally:server.server_close()

def run_local():
    check_environment()
    # Lock is released by the OS when the supervisor exits; no unsafe PID-based killing.
    with (RUNTIME/'launcher.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            for _ in range(40):
                if (RUNTIME/'url.txt').exists():
                    url=(RUNTIME/'url.txt').read_text().strip()
                    print('该项目已经运行：'+url,flush=True);browse(url);return
                time.sleep(.25)
            raise RuntimeError('该项目正在启动，请查看已经打开的启动终端。')
        (RUNTIME/'url.txt').unlink(missing_ok=True)
        port=free_port(CONFIG['port'])
        if CONFIG['mode']=='trip':
            cwd=ROOT/CONFIG['directory']
            cmd=[sys.executable,'-u',str(cwd/'demo_server.py'),'--host','127.0.0.1','--port',str(port)]
            suffix='/';health='/health'
        else:
            cwd=ROOT;cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--serve-portfolio',str(port)]
            suffix='/'+quote('打开作品集.html');health=suffix
        proc=subprocess.Popen(cmd,cwd=cwd,start_new_session=True)
        try:
            ready=False
            for _ in range(100):
                if proc.poll() is not None:raise RuntimeError('项目进程提前退出，请查看上方错误。')
                try:
                    with urlopen(f'http://127.0.0.1:{port}{health}',timeout=1) as r:ready=r.status==200
                    if ready:break
                except OSError:pass
                time.sleep(.1)
            if not ready:raise RuntimeError('服务未能在规定时间内就绪。')
            url=f'http://127.0.0.1:{port}{suffix}'
            (RUNTIME/'url.txt').write_text(url)
            print(f'\n启动成功：{url}\n请保留本终端窗口；按 Ctrl+C 停止该项目。',flush=True)
            browse(url)
            returncode=proc.wait()
            if returncode:raise RuntimeError(f'服务异常退出，退出码 {returncode}。')
        finally:
            (RUNTIME/'url.txt').unlink(missing_ok=True)
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--stop',action='store_true');parser.add_argument('--serve-portfolio',type=int)
    args=parser.parse_args()
    if args.serve_portfolio:serve_portfolio(args.serve_portfolio);return
    if args.check:check_environment();return
    if CONFIG['mode']=='docker':run_docker(args.stop)
    else:run_local()

if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    try:main()
    except KeyboardInterrupt:print('\n已退出本次启动。',flush=True)
    except Exception as exc:
        print(f'\n启动失败：{exc}',file=sys.stderr,flush=True);sys.exit(1)
