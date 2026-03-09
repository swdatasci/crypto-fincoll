module.exports = {
  apps: [
    {
      name: 'fincoll',
      script: '/home/rford/caelum/caelum-supersystem/fincoll/.venv/bin/python3',
      args: '-m uvicorn fincoll.server:app --host 0.0.0.0 --port 8002',
      cwd: '/home/rford/caelum/caelum-supersystem/fincoll',
      interpreter: 'none',
      env_file: '/home/rford/caelum/caelum-supersystem/fincoll/.env',
      env: {
        PYTHONPATH: '/home/rford/caelum/caelum-supersystem/finvec',
        VELOCITY_CHECKPOINT: '/home/rford/caelum/caelum-supersystem/finvec/checkpoints/velocity/best_model.pt',
        VELOCITY_DEVICE: 'cuda',
        NODE_ENV: 'production'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      error_file: '/home/rford/caelum/caelum-supersystem/fincoll/logs/pm2-error.log',
      out_file: '/home/rford/caelum/caelum-supersystem/fincoll/logs/pm2-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      min_uptime: '30s',
      max_restarts: 50,
      restart_delay: 35000,
      kill_timeout: 10000
    }
  ]
};
