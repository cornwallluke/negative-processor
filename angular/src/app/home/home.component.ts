import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { environment } from 'src/environments/environment';
import { albumMeta } from '../utils/album';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {

  albums:string[] = [];

  constructor(
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    const url = environment.api+"/picture/"
    this.http.get( url).subscribe((resp:any)=>{
      // console.log(resp);
      this.albums = resp;
      
    })
  }

}
